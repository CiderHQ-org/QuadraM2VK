// moltenvk-msl-helper: SPIRV-Cross's MSL conversion as MoltenVK 1.4.1 drives it.
//
// Built only by `scripts/moltenvk-msl-helper/build` against SPIRV-Cross at the commit MoltenVK
// 1.4.1 links; validation tooling, never part of the published crate. The configuration (every
// MSL option, resource binding, discrete descriptor set, and dynamic buffer) is derived by
// `crates/metal2vulkan-validation/src/msl_portability/moltenvk.rs` and written beside each module
// `corpus-pipeline --keep-modules` keeps; this program only applies
// it in the order MoltenVK's `SPIRVToMSLConverter::convert` does (MoltenVKShaderConverter
// `SPIRVToMSLConverter.cpp`) and reports what MoltenVK reads back for the Metal compile.
//
//   moltenvk-msl-helper --identity
//   moltenvk-msl-helper <config> <module.spv> <out.metal>
//
// Config lines (whitespace-separated; the entry name is the rest of its line):
//   entry <execution-model> <name>
//   common <flip_vert_y|fixup_clipspace> <0|1>
//   option <CompilerMSL::Options field> <unsigned value>
//   binding <execution-model> <void|image|sampler|sampled_image> <set> <binding> <count>
//           <msl_buffer> <msl_texture> <msl_sampler>
//   discrete <set>
//   dynamic <execution-model> <set> <binding> <index>
//
// Exit 0: the MSL is written and stdout holds `fp_fast_math_flags=<u32>` and
// `position_invariant=<0|1>`. Exit 1: SPIRV-Cross threw `CompilerError` (MoltenVK logs
// "SPIR-V to MSL conversion error: <what>" and fails the pipeline). Exit 2: a usage, input, or
// configuration error (a harness fault, never a verdict on the module). Exit 3: any other
// exception.

#include "spirv_msl.hpp"

#include "moltenvk_msl_helper_identity.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace SPIRV_CROSS_NAMESPACE;

namespace
{

[[noreturn]] void fail(int code, const std::string &message)
{
	std::fprintf(stderr, "%s\n", message.c_str());
	std::exit(code);
}

bool parse_u32(const std::string &text, uint32_t &out)
{
	if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos || text.size() > 10)
		return false;
	unsigned long long value = std::stoull(text);
	if (value > 0xffffffffull)
		return false;
	out = uint32_t(value);
	return true;
}

// Sets one `CompilerMSL::Options` field by name. Only the fields MoltenVK 1.4.1 assigns.
bool set_option(CompilerMSL::Options &o, const std::string &name, uint32_t value)
{
#define M2V_U32(field)          \
	if (name == #field)         \
	{                           \
		o.field = value;        \
		return true;            \
	}
#define M2V_BOOL(field)         \
	if (name == #field)         \
	{                           \
		if (value > 1)          \
			return false;       \
		o.field = value != 0;   \
		return true;            \
	}
	if (name == "platform")
	{
		if (value > uint32_t(CompilerMSL::Options::macOS))
			return false;
		o.platform = CompilerMSL::Options::Platform(value);
		return true;
	}
	if (name == "argument_buffers_tier")
	{
		if (value > uint32_t(CompilerMSL::Options::ArgumentBuffersTier::Tier2))
			return false;
		o.argument_buffers_tier = CompilerMSL::Options::ArgumentBuffersTier(value);
		return true;
	}
	M2V_U32(msl_version)
	M2V_U32(texel_buffer_texture_width)
	M2V_U32(r32ui_linear_texture_alignment)
	M2V_U32(buffer_size_buffer_index)
	M2V_U32(dynamic_offsets_buffer_index)
	M2V_U32(indirect_params_buffer_index)
	M2V_U32(shader_output_buffer_index)
	M2V_U32(view_mask_buffer_index)
	M2V_U32(fixed_subgroup_size)
	M2V_BOOL(texture_buffer_native)
	M2V_BOOL(dispatch_base)
	M2V_BOOL(texture_1D_as_2D)
	M2V_BOOL(argument_buffers)
	M2V_BOOL(force_active_argument_buffer_resources)
	M2V_BOOL(pad_argument_buffer_resources)
	M2V_BOOL(emulate_subgroups)
	M2V_BOOL(replace_recursive_inputs)
	M2V_BOOL(pad_fragment_output_components)
	M2V_BOOL(agx_manual_cube_grad_fixup)
	M2V_BOOL(ios_support_base_vertex_instance)
	M2V_BOOL(tess_domain_origin_lower_left)
	M2V_BOOL(multiview)
	M2V_BOOL(multiview_layered_rendering)
	M2V_BOOL(view_index_from_device_index)
	M2V_BOOL(capture_output_to_buffer)
	M2V_BOOL(disable_rasterization)
	M2V_BOOL(check_discarded_frag_stores)
	M2V_BOOL(force_fragment_with_side_effects_execution)
	M2V_BOOL(input_attachment_is_ds_attachment)
#undef M2V_U32
#undef M2V_BOOL
	return false;
}

bool parse_basetype(const std::string &name, SPIRType::BaseType &out)
{
	if (name == "void")
		out = SPIRType::Void;
	else if (name == "image")
		out = SPIRType::Image;
	else if (name == "sampler")
		out = SPIRType::Sampler;
	else if (name == "sampled_image")
		out = SPIRType::SampledImage;
	else
		return false;
	return true;
}

struct Dynamic
{
	spv::ExecutionModel stage;
	uint32_t set, binding, index;
};

struct Config
{
	std::string entry;
	spv::ExecutionModel stage = spv::ExecutionModelMax;
	bool flip_vert_y = false;
	bool fixup_clipspace = false;
	CompilerMSL::Options options;
	std::vector<MSLResourceBinding> bindings;
	std::vector<uint32_t> discrete;
	std::vector<Dynamic> dynamic;
};

Config read_config(const char *path)
{
	std::ifstream in(path);
	if (!in)
		fail(2, std::string("helper: cannot read config ") + path);
	// Every option starts at SPIRV-Cross's default; the config sets each one MoltenVK assigns.
	Config config;
	std::string line;
	uint32_t number = 0;
	while (std::getline(in, line))
	{
		number++;
		std::istringstream words(line);
		std::string directive;
		if (!(words >> directive))
			continue;
		auto bad = [&]() { fail(2, "helper: config line " + std::to_string(number) + ": " + line); };
		auto u32 = [&]() {
			std::string text;
			uint32_t value = 0;
			if (!(words >> text) || !parse_u32(text, value))
				bad();
			return value;
		};
		if (directive == "entry")
		{
			config.stage = spv::ExecutionModel(u32());
			std::string rest;
			std::getline(words >> std::ws, rest);
			if (rest.empty())
				bad();
			config.entry = rest;
			continue;
		}
		if (directive == "common")
		{
			std::string name;
			if (!(words >> name))
				bad();
			uint32_t value = u32();
			if (value > 1)
				bad();
			if (name == "flip_vert_y")
				config.flip_vert_y = value != 0;
			else if (name == "fixup_clipspace")
				config.fixup_clipspace = value != 0;
			else
				bad();
		}
		else if (directive == "option")
		{
			std::string name;
			if (!(words >> name))
				bad();
			if (!set_option(config.options, name, u32()))
				bad();
		}
		else if (directive == "binding")
		{
			MSLResourceBinding binding;
			binding.stage = spv::ExecutionModel(u32());
			std::string basetype;
			if (!(words >> basetype) || !parse_basetype(basetype, binding.basetype))
				bad();
			binding.desc_set = u32();
			binding.binding = u32();
			binding.count = u32();
			binding.msl_buffer = u32();
			binding.msl_texture = u32();
			binding.msl_sampler = u32();
			config.bindings.push_back(binding);
		}
		else if (directive == "discrete")
			config.discrete.push_back(u32());
		else if (directive == "dynamic")
		{
			Dynamic d;
			d.stage = spv::ExecutionModel(u32());
			d.set = u32();
			d.binding = u32();
			d.index = u32();
			config.dynamic.push_back(d);
		}
		else
			bad();
		std::string extra;
		if (words >> extra)
			bad();
	}
	if (config.entry.empty())
		fail(2, "helper: the config names no entry point");
	return config;
}

std::vector<uint32_t> read_spirv(const char *path)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
		fail(2, std::string("helper: cannot read ") + path);
	std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (bytes.empty() || bytes.size() % 4 != 0)
		fail(2, std::string("helper: ") + path + " is not a whole number of words");
	std::vector<uint32_t> words(bytes.size() / 4);
	std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char *>(words.data()));
	return words;
}

} // namespace

int main(int argc, char **argv)
{
	if (argc == 2 && std::string(argv[1]) == "--identity")
	{
		std::printf("moltenvk-msl-helper\n");
		std::printf("spirv_cross_commit=%s\n", M2V_SPIRV_CROSS_COMMIT);
		std::printf("helper_source_sha256=%s\n", M2V_HELPER_SOURCE_SHA256);
		std::printf("compiler=%s\n", M2V_HELPER_COMPILER);
		return 0;
	}
	if (argc != 4)
		fail(2, "usage: moltenvk-msl-helper --identity | <config> <module.spv> <out.metal>");
	Config config = read_config(argv[1]);
	std::vector<uint32_t> spirv = read_spirv(argv[2]);

	std::string msl;
	uint32_t fp_fast_math_flags = 0;
	bool position_invariant = false;
	try
	{
		// `SPIRVToMSLConverter::convert`, in its order.
		CompilerMSL compiler(std::move(spirv));
		compiler.set_entry_point(config.entry, config.stage);
		compiler.set_msl_options(config.options);
		auto common = compiler.get_common_options();
		common.vertex.flip_vert_y = config.flip_vert_y;
		common.vertex.fixup_clipspace = config.fixup_clipspace;
		compiler.set_common_options(common);
		for (auto &binding : config.bindings)
			if (binding.stage == config.stage)
				compiler.add_msl_resource_binding(binding);
		for (uint32_t set : config.discrete)
			compiler.add_discrete_descriptor_set(set);
		if (config.options.argument_buffers)
			for (auto &d : config.dynamic)
				if (d.stage == config.stage)
					compiler.add_dynamic_buffer(d.set, d.binding, d.index);
		msl = compiler.compile();
		// `populateEntryPoint` and the result info MoltenVK compiles the library with.
		fp_fast_math_flags = compiler.get_fp_fast_math_flags(true);
		position_invariant = compiler.is_position_invariant();
	}
	catch (const CompilerError &error)
	{
		fail(1, std::string("SPIR-V to MSL conversion error: ") + error.what());
	}
	catch (const std::exception &error)
	{
		fail(3, std::string("helper: unexpected exception: ") + error.what());
	}

	std::ofstream out(argv[3], std::ios::binary);
	out << msl;
	out.close();
	if (!out)
		fail(2, std::string("helper: cannot write ") + argv[3]);
	std::printf("fp_fast_math_flags=%u\n", fp_fast_math_flags);
	std::printf("position_invariant=%d\n", position_invariant ? 1 : 0);
	return 0;
}
