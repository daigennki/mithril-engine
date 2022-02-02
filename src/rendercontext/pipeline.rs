/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::io::Read;
use vulkano::render_pass::RenderPass;
use vulkano::pipeline::graphics::viewport;
use vulkano::render_pass::Subpass;

pub struct Pipeline
{
    pipeline: Arc<vulkano::pipeline::GraphicsPipeline>
}
impl Pipeline
{
    pub fn new(
        vk_dev: Arc<vulkano::device::Device>, 
        vs_filename: &str, 
        fs_filename: Option<&str>, 
        render_pass: Arc<RenderPass>, 
        width: u32, height: u32
    ) -> Result<Pipeline, Box<dyn std::error::Error>>
    {
        let rp_subpass = Subpass::from(render_pass.clone(), 0).ok_or("Subpass 0 for render pass doesn't exist!")?;

        let viewport = viewport::Viewport{ 
            origin: [ 0.0, 0.0 ],
            dimensions: [ width as f32, height as f32 ],
            depth_range: (-1.0..1.0)
        };

        // load vertex shader
        let vs = load_spirv(vk_dev.clone(), &vs_filename)?;
        let vs_entry = vs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;

        // do some building
        let mut pipeline = vulkano::pipeline::GraphicsPipeline::start()
            .viewport_state(viewport::ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            .vertex_shader(vs_entry, ())
            .render_pass(rp_subpass);

        // load fragment shader (optional)
        let fs;
        match fs_filename {
            Some(f) => {
                fs = load_spirv(vk_dev.clone(), &f)?;
                let fs_entry = fs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
                pipeline = pipeline.fragment_shader(fs_entry, ());
            }
            None => ()
        }
            
        Ok(Pipeline{
            pipeline: pipeline.build(vk_dev)?
        })
    }
}

fn load_spirv(device: Arc<vulkano::device::Device>, filename: &str) 
	-> Result<Arc<vulkano::shader::ShaderModule>, Box<dyn std::error::Error>>
{
	let mut spv_file = std::fs::File::open(filename)
		.or_else(|e| Err(format!("Failed to open SPIR-V shader file '{}': {}", filename, e)))?;

	let mut spv_data: Vec<u8> = Vec::new();
	spv_file.read_to_end(&mut spv_data)
		.or_else(|e| Err(format!("Failed to read SPIR-V shader file '{}': {}", filename, e)))?;

	Ok(unsafe { vulkano::shader::ShaderModule::from_bytes(device, &spv_data) }?)
}
