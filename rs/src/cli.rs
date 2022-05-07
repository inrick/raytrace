use getopts::Options;

use crate::ray::{raytrace, save_file, Args};

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
		"t",
		"threads",
		"number of threads to run on [default: 8]",
		"THREADS",
	);

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
	let threads: u32 = parsed
		.opt_str("t")
		.unwrap_or_else(|| "8".to_owned())
		.parse()
		.map_err(|_| "number of threads must be a positive number")?;
	if threads == 0 {
		return Err("number of threads must be a positive number".into());
	}

	let args = Args { nsamples, threads };

	let img = raytrace(&args, 600, 300);
	save_file(&img, &output)
}
