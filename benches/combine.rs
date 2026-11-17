use crc_fast::CrcAlgorithm::{Crc32Iscsi, Crc32IsoHdlc, Crc64Nvme};
use crc_fast::{checksum, checksum_combine, CrcAlgorithm, Digest};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const ALGORITHMS: [(&str, CrcAlgorithm, u64); 3] = [
    ("CRC-32/ISCSI", Crc32Iscsi, 0xffff_ffff),
    ("CRC-32/ISO-HDLC", Crc32IsoHdlc, 0xffff_ffff),
    ("CRC-64/NVME", Crc64Nvme, u64::MAX),
];

fn combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("combine");
    for (name, algorithm, _) in ALGORITHMS {
        for len in [4 << 10, 1_234_567, u64::MAX] {
            group.bench_function(BenchmarkId::new(name, len), |b| {
                b.iter(|| checksum_combine(algorithm, black_box(1), black_box(2), black_box(len)))
            });
        }
    }
}

fn fold(c: &mut Criterion) {
    let data: Vec<u8> = (0..1 << 20).map(|i| i as u8).collect();
    let mut group = c.benchmark_group("fold 1 MiB");
    group.throughput(Throughput::Bytes(data.len() as u64));
    for (name, algorithm, xorout) in ALGORITHMS {
        let empty = checksum(algorithm, &[]);
        for segment in [256, 4 << 10, 64 << 10] {
            group.bench_function(BenchmarkId::new(format!("{name} combine"), segment), |b| {
                b.iter(|| {
                    data.chunks(segment).fold(empty, |crc, chunk| {
                        let len = chunk.len() as u64;
                        checksum_combine(algorithm, crc, checksum(algorithm, chunk), len)
                    })
                })
            });
            group.bench_function(BenchmarkId::new(format!("{name} append"), segment), |b| {
                b.iter(|| {
                    data.chunks(segment).fold(empty, |crc, chunk| {
                        let mut digest = Digest::new_with_init_state(algorithm, crc ^ xorout);
                        digest.update(chunk);
                        digest.finalize()
                    })
                })
            });
        }
    }
}

criterion_group!(benches, combine, fold);
criterion_main!(benches);
