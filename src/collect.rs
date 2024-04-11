use crate::{items, operators};
use clap::Args;
use std::{
    collections::HashSet,
    fs::{self, File},
    io::Write,
    mem::take,
    path::PathBuf,
};

#[derive(Args, Debug)]
pub struct Condition {
    /// Filter by operator name
    #[clap(long)]
    op: Option<Vec<String>>,
    /// Filter by domain item
    #[clap(long, short)]
    domain: Option<Vec<String>>,
    /// Output directory
    #[clap(long, short)]
    target: Option<PathBuf>,
}

impl Condition {
    pub fn filter(self) -> std::io::Result<()> {
        let mut op = self.op.unwrap_or_default();
        if op.len() > 1 {
            op = op.into_iter().collect::<HashSet<_>>().into_iter().collect();
        }

        let domain = self
            .domain
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<_>>();

        let collector = match &mut *op {
            [] => todo!(),
            [op] => Collector::op(take(op), domain)?,
            _ => todo!(),
        };
        collector.write(self.target)
    }
}

struct Collector {
    name: String,
    title: String,
    files: Vec<PathBuf>,
}

impl Collector {
    fn write(self, target: Option<PathBuf>) -> std::io::Result<()> {
        if !self.files.is_empty() {
            if let Some(target) = target {
                if target.is_file() {
                    todo!()
                } else if target.is_dir() {
                    todo!()
                } else if target.exists() {
                    todo!()
                } else {
                    fs::create_dir_all(&target)?;
                    let mut file = File::create(&target.join(self.name))?;
                    concat(&mut file, &self.title, &self.files).unwrap();
                }
            } else {
                let mut file = File::create(self.name)?;
                concat(&mut file, &self.title, &self.files).unwrap();
            }
        } else {
            println!("No item found");
        }
        Ok(())
    }

    fn op(target: String, domain: HashSet<String>) -> std::io::Result<Self> {
        let operators = operators();
        let items = items()?
            .filter(|item| {
                item.op == target
                    && (domain.is_empty() || item.domain.split('-').any(|s| domain.contains(s)))
            })
            .map(|item| operators.join(format!("{}.{}.md", item.op, item.domain)));

        Ok(Self {
            name: format!("{target}.md"),
            files: items.collect(),
            title: target,
        })
    }
}

fn concat(write: &mut impl Write, title: &str, files: &[PathBuf]) -> std::io::Result<()> {
    writeln!(write, "# {title}")?;
    let mut blank = false;
    for file in files {
        let content = fs::read_to_string(file)?;
        let content = if let Some(s) = content.as_bytes().strip_prefix(&[0xef, 0xbb, 0xbf]) {
            unsafe { std::str::from_utf8_unchecked(s) }
        } else {
            &content
        };

        if !blank {
            writeln!(write)?;
            blank = true;
        }

        for line in content.lines() {
            let line = line.trim();
            blank = line.is_empty();
            if line.starts_with('#') {
                writeln!(write, "#{line}")?;
            } else {
                writeln!(write, "{line}")?;
            }
        }
    }
    Ok(())
}
