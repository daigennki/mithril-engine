/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::path::Path;

fn main()
{
	// Compile HLSL shaders in src/shaders into SPIR-V.
	let compiler = shaderc::Compiler::new().unwrap();
	let mut options = shaderc::CompileOptions::new().unwrap();
	options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
	options.set_source_language(shaderc::SourceLanguage::HLSL);
	options.set_include_callback(shader_src_include_callback);
	options.set_hlsl_io_mapping(true);

	let shader_paths = std::fs::read_dir("./src/shaders")
		.unwrap()
		.map(|d| d.unwrap().path())
		.filter(|p| p.extension().unwrap_or(std::ffi::OsStr::new("")) == "hlsl");

	for shader_path in shader_paths {
		match detect_shader_stage(&shader_path) {
			Some(shader_stage) => compile_shader(&shader_path, shader_stage, &compiler, &options),
			None => println!(
				"could not detect shader stage for shader '{}', skipping...",
				shader_path.to_str().unwrap()
			),
		}
	}

	// This will only run if files in the src/shaders directory have been changed,
	// or if this build script has been changed.
	println!("cargo:rerun-if-changed=src/shaders/");
	println!("cargo:rerun-if-changed=build.rs");
}

fn compile_shader(path: &Path, stage: shaderc::ShaderKind, compiler: &shaderc::Compiler, options: &shaderc::CompileOptions)
{
	let shader_src = String::from_utf8(std::fs::read(&path).unwrap()).unwrap();
	let shader_file_name = path.file_name().unwrap().to_str().unwrap();

	let compile_result = compiler
		.compile_into_spirv(&shader_src, stage, shader_file_name, "main", Some(&options))
		.unwrap();

	let spv_file_name = format!("{}.spv", path.file_stem().unwrap().to_str().unwrap());
	let output_path = Path::new("./shaders/").join(spv_file_name);
	//let output_path = Path::new(&std::env::var("OUT_DIR").unwrap()).join(spv_file_name);
	std::fs::write(&output_path, compile_result.as_binary_u8()).unwrap();
	println!("wrote {}", output_path.to_str().unwrap());
}

fn shader_src_include_callback(
	src_req: &str,
	_include_type: shaderc::IncludeType,
	src_containing: &str,
	_include_depth: usize,
) -> shaderc::IncludeCallbackResult
{
	let src_req_path = Path::new("./src/shaders/").join(src_req);
	let content = String::from_utf8(std::fs::read(&src_req_path).expect(&format!(
		"failed to read shader source '{}' included from '{}'",
		src_req, src_containing
	)))
	.unwrap();

	Ok(shaderc::ResolvedInclude {
		resolved_name: src_req_path.to_str().unwrap().to_string(),
		content,
	})
}

fn detect_shader_stage(path: &Path) -> Option<shaderc::ShaderKind>
{
	let path_str = path.to_str().unwrap();
	if path_str.ends_with(".vert.hlsl") {
		Some(shaderc::ShaderKind::Vertex)
	} else if path_str.ends_with(".frag.hlsl") {
		Some(shaderc::ShaderKind::Fragment)
	} else {
		None
	}
}
