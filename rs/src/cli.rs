use getopts::Options;
use std::time::Instant;

use crate::ray::{
	camera_default, raytrace, save_file, small_scene_moving, Args, Config,
};

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

	let program_args: Vec<String> = std::env::args().collect();
	let program = program_args[0].clone();
	let parsed = opts.parse(&program_args[1..])?;
	if parsed.opt_present("h") {
		print_usage(&opts, &program, 0);
	}

	let output = parsed.opt_str("o").unwrap_or_else(|| "out.png".to_owned());
	let nsamples: u32 = parsed
		.opt_str("n")
		.unwrap_or_else(|| "10".to_owned())
		.parse()
		.map_err(|_| "number of samples must be a positive number")?;
	if nsamples == 0 {
		return Err("number of samples must be a positive number".into());
	}
	let max_depth: u32 = parsed
		.opt_str("d")
		.unwrap_or_else(|| "50".to_owned())
		.parse()
		.map_err(|_| "depth must be a positive number")?;
	if max_depth == 0 {
		return Err("depth must be a positive number".into());
	}
	let threads: u32 = parsed
		.opt_str("t")
		.unwrap_or_else(|| "8".to_owned())
		.parse()
		.map_err(|_| "number of threads must be a positive number")?;
	if threads == 0 {
		return Err("number of threads must be a positive number".into());
	}
	let nx: u32 = parsed
		.opt_str("x")
		.unwrap_or_else(|| "600".to_owned())
		.parse()
		.map_err(|_| "image width must be a positive number")?;
	if nx == 0 {
		return Err("image width must be a positive number".into());
	}
	let ny: u32 = parsed
		.opt_str("y")
		.unwrap_or_else(|| "300".to_owned())
		.parse()
		.map_err(|_| "image height must be a positive number")?;
	if ny == 0 {
		return Err("image height must be a positive number".into());
	}

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

	save_file(&img, &output)
}
