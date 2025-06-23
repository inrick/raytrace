use getopts::Options;
use std::num::NonZero;
use std::time::Instant;

use crate::ray::{camera_default, raytrace, save_file, Args, Config};
use crate::scene::small_scene_moving;

type Error = Box<dyn std::error::Error>;
type Result<T> = ::std::result::Result<T, Error>;

fn print_usage(opts: &Options, program: &str, exit_code: i32) -> ! {
	let usage = format!("Usage:\n    {} [OPTIONS]", program);
	println!("{}", opts.usage(&usage));
	std::process::exit(exit_code);
}

pub fn run() -> Result<()> {
	let mut opts = Options::new();
	opts.optflag("h", "help", "show help");
	opts.optopt(
		"o",
		"output",
		"output file name (supports .ppm/.png/.jpg) [default: \"out.png\"]",
		"NAME",
	);
	opts.optopt(
		"n",
		"nsamples",
		"number of samples per ray [default: 10]",
		"SAMPLES",
	);
	opts.optopt(
		"d",
		"depth",
		"maximum number of ray bounces [default: 50]",
		"SAMPLES",
	);
	opts.optopt(
		"t",
		"threads",
		"number of threads to run on [default: 8]",
		"THREADS",
	);
	opts.optopt("x", "width", "width of image [default: 600]", "PIXELS");
	opts.optopt("y", "height", "height of image [default: 300]", "PIXELS");
	pprof::register_flag(&mut opts);

	let program_args: Vec<String> = std::env::args().collect();
	let program = program_args[0].clone();
	let parsed = opts.parse(&program_args[1..])?;
	if parsed.opt_present("h") {
		print_usage(&opts, &program, 0);
	}

	let output = parsed.opt_str("o").unwrap_or_else(|| "out.png".to_owned());
	let nsamples: NonZero<u32> = parsed
		.opt_str("n")
		.unwrap_or_else(|| "10".to_owned())
		.parse()
		.map_err(|_| "number of samples must be a positive number")?;
	let max_depth: NonZero<u32> = parsed
		.opt_str("d")
		.unwrap_or_else(|| "50".to_owned())
		.parse()
		.map_err(|_| "depth must be a positive number")?;
	let threads: u32 = parsed
		.opt_str("t")
		.unwrap_or_else(|| "0".to_owned())
		.parse()
		.map_err(|_| "number of threads must be a non-negative number")?;
	let nx: NonZero<u32> = parsed
		.opt_str("x")
		.unwrap_or_else(|| "600".to_owned())
		.parse()
		.map_err(|_| "image width must be a positive number")?;
	let ny: NonZero<u32> = parsed
		.opt_str("y")
		.unwrap_or_else(|| "300".to_owned())
		.parse()
		.map_err(|_| "image height must be a positive number")?;

	let profiler = pprof::start_cpu_profile(&parsed)?;

	let t0 = Instant::now();

	let cfg = Config {
		nsamples,
		max_depth,
		threads,
		nx,
		ny,
	};

	let cam = camera_default(&cfg);
	let scene = small_scene_moving();
	let args = Args { cfg, cam, scene };

	let img = raytrace(&args);

	let t1 = Instant::now();
	println!(
		"rendering took {:.3} seconds",
		t1.duration_since(t0).as_secs_f32()
	);

	pprof::write_cpu_profile(profiler);

	save_file(&img, &output)
}

#[cfg(feature = "pprof")]
mod pprof {
	use super::Result;

	use pprof::protos::Message;
	use pprof::ProfilerGuard;
	use std::fs::File;
	use std::io::Write;

	pub struct ProfileOptions<'a> {
		filename: String,
		guard: ProfilerGuard<'a>,
	}

	pub fn register_flag(opts: &mut getopts::Options) {
		opts.optopt("", "cpuprof", "path where to write cpu profile", "");
	}

	pub fn start_cpu_profile(
		parsed: &getopts::Matches,
	) -> Result<Option<ProfileOptions>> {
		let filename: String = parsed
			.opt_str("cpuprof")
			.unwrap_or_else(|| "".to_owned())
			.parse()
			.map_err(|_| "need a valid file name")?;
		if filename != "" {
			let guard = pprof::ProfilerGuardBuilder::default()
				.frequency(1000)
				.blocklist(&["libc", "libgcc", "pthread", "vdso"])
				.build()?;
			Ok(Some(ProfileOptions { filename, guard }))
		} else {
			Ok(None)
		}
	}

	pub fn write_cpu_profile(opts: Option<ProfileOptions>) {
		if let Some(opts) = opts {
			match write_report(&opts.filename, opts.guard) {
				Ok(_) => println!("cpu profile written to {}", &opts.filename),
				Err(err) => println!("error writing cpu profile: {}", err),
			};
		}
	}

	fn write_report(filename: &str, guard: ProfilerGuard) -> Result<()> {
		let report = guard.report().build()?;
		let mut file = File::create(filename)?;
		let profile = report.pprof()?;
		let mut content = Vec::new();
		profile.encode(&mut content)?;
		file.write_all(&content)?;
		Ok(())
	}
}

#[cfg(not(feature = "pprof"))]
mod pprof {
	use super::Result;

	pub fn register_flag(_: &mut getopts::Options) {}

	pub fn start_cpu_profile(_: &getopts::Matches) -> Result<Option<()>> {
		Ok(None)
	}

	pub fn write_cpu_profile(_: Option<()>) {}
}
