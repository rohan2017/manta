# Preintegrated IMU packet contract

`manta.PreintegratedIMUPacket` wraps preintegrator readouts or parsed device
fields. `filter.predict(packet)` consumes it through the ordinary Manta filter
runtime. The INS must select `propagation="preintegrated"` and the matching
accelerometer integration mode. Plain mapping packets and
`predict_preintegrated(...)` remain supported.

## Values

`manta.estimation.imu_packet.PACKET_LAYOUT` defines the canonical field order
and shapes. This is also the flattened generated preintegrator readout order;
`PACKET_FIELDS` is derived from it. Important conventions:

- Quaternion: scalar-first `[w,x,y,z]`, rotating vectors from end-IMU axes into start-IMU axes.
- Delta velocity/position: m/s and metres, in the IMU frame at packet start, before gravity and planet rotation compensation.
- Delta covariance: 9×9, errors ordered rotation vector, velocity, position.
- Bias Jacobian: 9×6, columns gyro bias (rad/s), accelerometer bias (m/s²).
- Bias reference: the values used during preintegration; host prediction applies first-order bias correction.
- Boundary cross-covariances: with **standardized** acquisition errors. The corresponding physical noise sigmas are separate fields.
- Duration: seconds; sample count: number of integrated intervals.
- `accel_endpoint`: 1 for point endpoint integration, 0 for left-held acceleration.
- Matrices: column-major (Fortran/CasADi) flattening. A two-dimensional array is also accepted by the Python constructor.

The class validates shape, finite values, duration, count, mode, quaternion,
and covariance. It owns immutable copies. Typed prediction checks that the
packet starts at `filter.time`, refusing duplicates or gaps before mutation.
Use `reset(..., t=epoch)` to initialize state at a nonzero epoch.
Clock domain, sensor identity, calibration identity, CRC, loss recovery, and
transport framing remain application responsibilities.

## Firmware loop

The [INS handoff](https://github.com/Ulysses-Ecosystem-Engineering/ins-handoff)
runner's `--export-preintegrator` option exports `TargetCpp` artifacts,
including the C kernel source/header, binary-codec `imu_packet.h`, and
`imu_preintegrator.h`. The latter is a small C99 calling shim generated from
the module's actual state defaults and input layout. It owns state and
workspace in an `ImuPreintegrator` struct, with no heap allocation or Eigen.
Compile your firmware with `preintegrator_kernels.c` and the math library.
The Python exporter lives in handoff `run_ins.py` so it is easy to inspect or adapt.

Encode the readouts directly into the standard binary payload:

```c
static ImuPreintegrator imu;  /* state and work buffers: allocate once */
double inputs[MANTA_IMU_INPUT_COUNT] = {0};
imu_preintegrator_reset(&imu);
/* Fill ACCEL, GYRO, END_ACCEL, END_GYRO and bias references using the
   MANTA_IMU_INPUT_* offsets. Then call once per observed interval: */
int status = imu_preintegrator_step(&imu, inputs, dt, interval_start_time);
/* Handle nonzero status before transmitting. */
static uint8_t payload[MANTA_IMU_PACKET_SIZE];
manta_imu_packet_encode(payload, packet_start_time, imu.y);
/* Your transport sends exactly sizeof(payload) bytes. */
```

Keep the actual previous IMU acquisition. For each fresh pair, integrate the
interval using both endpoint samples and their actual timestamp difference.
At the desired boundary, send the packet and reset **preintegrator accumulated
state**, retaining the last raw acquisition as the next interval's left end.
Do not synthesize a duplicate sample or replace the shared boundary with an
independent one. The handoff raw-IMU runner shows the same endpoint handling.

Gyro/accelerometer densities are converted with the declared acquisition rate
into per-sample sigmas. These are independent point acquisitions; filtered
samples, hardware delta angles, and delta velocities require their own
acquisition contract. The first sample establishes an endpoint and cannot
produce a positive-duration packet by itself.

Close a packet at each aiding timestamp the host needs. A packet cannot be
split exactly after transmission from its aggregate deltas alone. Out-of-order
measurements, missing packets, and clock synchronization need an explicit
application policy; the packet class does not invent motion or observations.

## Standard binary encoding

`to_bytes()` / `from_bytes()` use a fixed **2,488-byte** schema-4 encoding:

| Offset | Type | Meaning |
|---|---|---|
| 0 | 4 bytes | ASCII `MIPK` |
| 4 | little-endian uint32 | schema = 4 |
| 8 | little-endian IEEE-754 float64 | start time in filter-clock seconds |
| 16 | 309 little-endian float64 values | `PACKET_LAYOUT`, without padding |

Do not serialize a native C struct and assume matching padding or endianness.
`PreintegratedIMUPacket.c_header()` emits this encoder and the named C struct.
The handoff exporter uses this codec by default. The host calls
`PreintegratedIMUPacket.from_bytes(payload)` on one complete payload. This
removes manual field packing from the application; the transport sees bytes.
At 50 packets/s this full covariance format uses 124,400 bytes/s before
transport overhead. Choose packet cadence and transport accordingly; dropping
cross-covariance fields is not a valid bandwidth optimization.

## Transport boundary

The codec is independent of UART, TCP, UDP or another link. Byte-stream reads
and writes can be partial or combine multiple payloads. The application must
buffer/reassemble packets, establish boundaries, and handle disconnection or
loss; message/frame size limits may require fragmentation. On an already
aligned reliable stream, consume `PreintegratedIMUPacket.wire_size` bytes per
payload. Do not decode arbitrary read chunks or rely on a single write/read
transferring the whole packet. Retain C payload buffers until asynchronous
transmission finishes.

Magic, schema, length and numerical checks are not a checksum. The codec adds
no CRC, authentication or automatic stream resynchronization; the transport
supplies the framing and integrity appropriate to the link. Unknown schemas
are rejected explicitly instead of silently interpreting incompatible fields.

## Optional custom serialization

`PreintegratedIMUPacket.c_struct_header()` generates a C99/C++ header containing
`MantaPreintegratedIMUPacket` and `manta_imu_packet_from_readouts(...)`. The
struct contains `uint32_t schema`, `double start_time`, and every field in
`PACKET_LAYOUT` as a named double or fixed-size double array. Matrices are
flattened column-major; this is a data contract, not a packed wire struct.
The copy function maps codegen readouts into those members without casting
flat arrays to structs or relying on native padding.

Use this path only when integrating with an existing encoding or another
specific protocol constraint. In that case the application owns serialization
and deserialization, then constructs:

```python
packet = PreintegratedIMUPacket(decoded_fields, start_time=decoded_start_time,
                               schema=decoded_schema)
filter.predict(packet)
```

Serialize all numerical fields and retain their precision and correlations.
Do not transmit native struct memory and assume matching padding/endianness.
Wire encoding is independent of the physical bus: the same encoding can be
transported over UART or Ethernet, and either bus can carry different encodings.
