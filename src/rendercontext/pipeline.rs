/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::io::Read;
use vulkano::pipeline::graphics::viewport::*;
use vulkano::shader::ShaderModule;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::RenderPass;
use vulkano::render_pass::Subpass;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::vertex_input::VertexInputRate;
use vulkano::pipeline::graphics::vertex_input::VertexInputBindingDescription;
use vulkano::pipeline::graphics::vertex_input::VertexInputAttributeDescription;
use vulkano::format::Format;

pub struct Pipeline
{
    vertex_input_state: VertexInputState,
    vs: Arc<ShaderModule>,
    fs: Option<Arc<ShaderModule>>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>
}
impl Pipeline
{
    pub fn new(
        vk_dev: Arc<vulkano::device::Device>, 
        vertex_input: impl IntoIterator<Item = (u32, Format)>,
        vs_filename: String, 
        fs_filename: Option<String>,
        render_pass: Arc<RenderPass>, 
        width: u32, height: u32
    ) -> Result<Pipeline, Box<dyn std::error::Error>>
    {
        let rp_subpass = Subpass::from(render_pass.clone(), 0).ok_or("Subpass 0 for render pass doesn't exist!")?;
        let viewport = Viewport{ 
            origin: [ 0.0, 0.0 ],
            dimensions: [ width as f32, height as f32 ],
            depth_range: (0.0..1.0)
        };

        // Automatically generate vertex input state from input formats and strides.
        // This assumes bindings in order starting at 0 and with 0 offset - the typical use case.
        // VertexInputState will remain as default if vertex_input was empty.
        let mut i: u32 = 0;
        let mut vertex_input_state = VertexInputState::new();
        for (stride, vertex_format) in vertex_input {
            vertex_input_state = vertex_input_state
                .binding(i, VertexInputBindingDescription{ stride: stride, input_rate: VertexInputRate::Vertex })
                .attribute(i, VertexInputAttributeDescription{ binding: 0, format: vertex_format, offset: 0 });
            i += 1;
        }

        // load vertex shader
        log::info!("Loading vertex shader {}...", &vs_filename);
        let vs = load_spirv(vk_dev.clone(), &format!("shaders/{}", &vs_filename))?;
        let vs_entry = vs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;

        // do some building
        let mut pipeline = GraphicsPipeline::start()
            .vertex_input_state(vertex_input_state.clone())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            .vertex_shader(vs_entry, ())
            .render_pass(rp_subpass);

        // load fragment shader (optional)
        let fs;
        let fs_optional = match &fs_filename {
            Some(f) => {
                log::info!("Loading fragment shader {}...", f);
                fs = load_spirv(vk_dev.clone(), &format!("shaders/{}", f))?;
                let fs_entry = fs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
                pipeline = pipeline.fragment_shader(fs_entry, ());
                Some(fs.clone())
            }
            None => None
        };

        let pipeline_built = pipeline.build(vk_dev)?;
            
        Ok(Pipeline{
            vertex_input_state: vertex_input_state,
            vs: vs,
            fs: fs_optional,
            render_pass: render_pass,
            pipeline: pipeline_built
        })
    }

    pub fn resize_viewport(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>>
    {
        let rp_subpass = Subpass::from(self.render_pass.clone(), 0).ok_or("Subpass 0 for render pass doesn't exist!")?;
        let viewport = Viewport{ 
            origin: [ 0.0, 0.0 ],
            dimensions: [ width as f32, height as f32 ],
            depth_range: (0.0..1.0)
        };

        let vs_entry = self.vs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
        let mut pipeline_builder = GraphicsPipeline::start()
            .vertex_input_state(self.vertex_input_state.clone())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            .vertex_shader(vs_entry, ())
            .render_pass(rp_subpass);
        match &self.fs {
            Some(fs) => {
                let fs_entry = fs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
                pipeline_builder = pipeline_builder.fragment_shader(fs_entry, ());
            }
            None => ()
        }

        self.pipeline = pipeline_builder.build(self.pipeline.device().clone())?;
        Ok(())
    }
}

fn load_spirv(device: Arc<vulkano::device::Device>, filename: &str) 
	-> Result<Arc<vulkano::shader::ShaderModule>, Box<dyn std::error::Error>>
{
	let mut spv_file = std::fs::File::open(filename)
		.or_else(|e| Err(format!("Failed to open '{}': {}", filename, e)))?;

	let mut spv_data: Vec<u8> = Vec::new();
	spv_file.read_to_end(&mut spv_data)
		.or_else(|e| Err(format!("Failed to read '{}': {}", filename, e)))?;

	Ok(unsafe { vulkano::shader::ShaderModule::from_bytes(device, &spv_data) }?)
}
