// ============================================================================
// state_io.rs — EvoLenia v2
// Binary snapshot save/load for headless->GUI workflows.
// ============================================================================

use std::fs::File;
use std::io::{self, Read, Write};

use crate::world::{BufferSnapshot, WORLD_HEIGHT, WORLD_WIDTH};

const MAGIC: &[u8; 8] = b"EVOSNP01";

pub fn save_snapshot(path: &str, snapshot: &BufferSnapshot) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&WORLD_WIDTH.to_le_bytes())?;
    file.write_all(&WORLD_HEIGHT.to_le_bytes())?;

    write_vec_f32(&mut file, &snapshot.mass)?;
    write_vec_f32(&mut file, &snapshot.energy)?;
    write_vec_f32(&mut file, &snapshot.genome_a)?;
    write_vec_f32(&mut file, &snapshot.genome_b)?;
    write_vec_f32(&mut file, &snapshot.resource)?;
    Ok(())
}

pub fn load_snapshot(path: &str) -> io::Result<BufferSnapshot> {
    let mut file = File::open(path)?;

    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid snapshot magic"));
    }

    let width = read_u32(&mut file)?;
    let height = read_u32(&mut file)?;
    if width != WORLD_WIDTH || height != WORLD_HEIGHT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "snapshot dimensions {}x{} incompatible with current world {}x{}",
                width, height, WORLD_WIDTH, WORLD_HEIGHT
            ),
        ));
    }

    let mass = read_vec_f32(&mut file)?;
    let energy = read_vec_f32(&mut file)?;
    let genome_a = read_vec_f32(&mut file)?;
    let genome_b = read_vec_f32(&mut file)?;
    let resource = read_vec_f32(&mut file)?;

    Ok(BufferSnapshot {
        mass,
        energy,
        genome_a,
        genome_b,
        resource,
    })
}

fn write_vec_f32(file: &mut File, values: &[f32]) -> io::Result<()> {
    let len = values.len() as u64;
    file.write_all(&len.to_le_bytes())?;
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_vec_f32(file: &mut File) -> io::Result<Vec<f32>> {
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    file.read_exact(&mut bytes)?;
    let mut values = Vec::with_capacity(len);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn read_u32(file: &mut File) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
