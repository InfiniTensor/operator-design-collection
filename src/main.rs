mod collect;

use clap::{Parser, Subcommand};
use collect::Condition;
use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

fn main() -> std::io::Result<()> {
    use Commands::*;
    match Cli::parse().command {
        Collect(condition) => condition.filter(),
    }
}

#[derive(Parser)]
#[clap(name = "transformer-utils")]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Filter reports based on certain conditions
    Collect(Condition),
}

fn operators() -> &'static PathBuf {
    static OPERATORS: OnceLock<PathBuf> = OnceLock::new();
    OPERATORS.get_or_init(|| Path::new(std::env!("CARGO_MANIFEST_DIR")).join("operators"))
}

struct Item {
    op: String,
    domain: String,
}

fn items() -> std::io::Result<impl Iterator<Item = Item>> {
    Ok(fs::read_dir(operators())?
        .filter_map(|result| result.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .map_or(false, |ext| ext == OsStr::new("md"))
        })
        .map(|path| {
            let name = path.file_stem().unwrap().to_str().unwrap();
            let (op, domain) = name.split_once('.').unwrap();
            Item {
                op: op.to_string(),
                domain: domain.to_string(),
            }
        }))
}
