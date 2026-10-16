// The probe that recorded metal-answers.txt. Owned code, written for this repository. Build it with
// `xcrun swiftc -O probe.swift -o probe`, build each fixture into <dir>/<fixture>.metallib with the
// compiler driver (`xcrun metal -x metal - -o <dir>/<fixture>.metallib < <fixture>.metal`, which
// writes the reflection Metal's library answers from; `metallib` of `.air` files does not), then
// run one case per process:
//   probe <dir> list     prints every case name
//   probe <dir> <case>   prints the case's lines
// metal-answers.txt is the cases' outputs in `list` order. Pipeline creation only; nothing is
// encoded or committed to the GPU.
import Foundation
import Metal

func fail(_ message: String) -> Never { print("error \(message)"); exit(2) }
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let args = CommandLine.arguments
guard args.count == 3 else { fail("usage: probe <dir> <case>") }
let dir = args[1]
var libraries: [String: MTLLibrary] = [:]
func library(_ fixture: String) -> MTLLibrary {
    if let l = libraries[fixture] { return l }
    do {
        let l = try device.makeLibrary(URL: URL(fileURLWithPath: "\(dir)/\(fixture).metallib"))
        libraries[fixture] = l
        return l
    } catch { fail("library \(fixture): \(error)") }
}
// An Objective-C getter by selector, where the Swift name is unsure.
func objc<T>(_ o: AnyObject, _ selector: String) -> T? {
    let sel = NSSelectorFromString(selector)
    guard o.responds(to: sel) else { return nil }
    return o.perform(sel)?.takeUnretainedValue() as? T
}
func structDesc(_ s: MTLStructType?, _ depth: Int) -> String {
    guard let s = s else { return "nil" }
    if depth > 6 { return "{...}" }
    return "{" + s.members.map { m -> String in
        var p = ["name=\(m.name)", "offset=\(m.offset)", "dataType=\(m.dataType.rawValue)", "argumentIndex=\(m.argumentIndex)"]
        if let st = m.structType() { p.append("struct=" + structDesc(st, depth + 1)) }
        if let at = m.arrayType() { p.append("array=" + arrayDesc(at, depth + 1)) }
        if let tr = m.textureReferenceType() { p.append("texref=" + texDesc(tr)) }
        if let pt = m.pointerType() { p.append("pointer=" + ptrDesc(pt, depth + 1)) }
        return "(" + p.joined(separator: " ") + ")"
    }.joined(separator: ",") + "}"
}
func arrayDesc(_ a: MTLArrayType, _ depth: Int) -> String {
    var p = ["elementType=\(a.elementType.rawValue)", "length=\(a.arrayLength)", "stride=\(a.stride)", "argumentIndexStride=\(a.argumentIndexStride)"]
    if let s = a.elementStructType() { p.append("struct=" + structDesc(s, depth + 1)) }
    if let e: MTLArrayType = objc(a, "elementArrayType") { p.append("array=" + arrayDesc(e, depth + 1)) }
    if let t = a.elementTextureReferenceType() { p.append("texref=" + texDesc(t)) }
    if let q = a.elementPointerType() { p.append("pointer=" + ptrDesc(q, depth + 1)) }
    return "[" + p.joined(separator: " ") + "]"
}
func texDesc(_ t: MTLTextureReferenceType) -> String {
    "<dataType=\(t.textureDataType.rawValue) type=\(t.textureType.rawValue) access=\(t.access.rawValue) depth=\(t.isDepthTexture)>"
}
func ptrDesc(_ p: MTLPointerType, _ depth: Int) -> String {
    var q = ["elementType=\(p.elementType.rawValue)", "access=\(p.access.rawValue)", "alignment=\(p.alignment)", "dataSize=\(p.dataSize)", "argumentBuffer=\(p.elementIsArgumentBuffer)"]
    if let s = p.elementStructType() { q.append("struct=" + structDesc(s, depth + 1)) }
    if let a: MTLArrayType = objc(p, "elementArrayType") { q.append("array=" + arrayDesc(a, depth + 1)) }
    return "*<" + q.joined(separator: " ") + ">"
}
func bindingDesc(_ b: MTLBinding) -> String {
    var p = ["name=\(b.name)", "type=\(b.type.rawValue)", "access=\(b.access.rawValue)", "index=\(b.index)", "used=\(b.isUsed)", "argument=\(b.isArgument)"]
    if let x = b as? MTLBufferBinding {
        p += ["bufferAlignment=\(x.bufferAlignment)", "bufferDataSize=\(x.bufferDataSize)", "bufferDataType=\(x.bufferDataType.rawValue)",
              "bufferStructType=" + structDesc(x.bufferStructType, 0),
              "bufferPointerType=" + (x.bufferPointerType.map { ptrDesc($0, 0) } ?? "nil")]
    }
    if let x = b as? MTLTextureBinding {
        p += ["textureType=\(x.textureType.rawValue)", "textureDataType=\(x.textureDataType.rawValue)", "depth=\(x.isDepthTexture)", "arrayLength=\(x.arrayLength)"]
    }
    if let x = b as? MTLThreadgroupBinding {
        p += ["threadgroupMemoryAlignment=\(x.threadgroupMemoryAlignment)", "threadgroupMemoryDataSize=\(x.threadgroupMemoryDataSize)"]
    }
    if let x = b as? MTLObjectPayloadBinding {
        p += ["objectPayloadAlignment=\(x.objectPayloadAlignment)", "objectPayloadDataSize=\(x.objectPayloadDataSize)"]
    }
    return p.joined(separator: " ")
}
func lines(_ label: String, _ bindings: [MTLBinding]) -> [String] {
    ["\(label) count=\(bindings.count)"] + bindings.map { "  " + bindingDesc($0) }
}
func function(_ fixture: String, _ name: String, _ constants: MTLFunctionConstantValues? = nil) -> MTLFunction {
    let l = library(fixture)
    if let c = constants {
        do { return try l.makeFunction(name: name, constantValues: c) } catch { fail("function \(name): \(error)") }
    }
    guard let f = l.makeFunction(name: name) else { fail("no function \(name)") }
    return f
}
func attribute(_ kind: String, _ a: MTLAttribute) -> String {
    "  \(kind) name=\(a.name) index=\(a.attributeIndex) type=\(a.attributeType.rawValue) active=\(a.isActive) patch=\(a.isPatchData) controlPoint=\(a.isPatchControlPointData)"
}
// A function's MTLFunction properties.
func properties(_ f: MTLFunction) {
    print("function \(f.name) functionType=\(f.functionType.rawValue) patchType=\(f.patchType.rawValue) patchControlPointCount=\(f.patchControlPointCount)")
    if let va = f.vertexAttributes {
        print("vertexAttributes count=\(va.count)")
        for a in va { print("  vertexAttribute name=\(a.name ?? "nil") index=\(a.attributeIndex) type=\(a.attributeType.rawValue) active=\(a.isActive) patch=\(a.isPatchData) controlPoint=\(a.isPatchControlPointData)") }
    } else { print("vertexAttributes nil") }
    if let sa = f.stageInputAttributes {
        print("stageInputAttributes count=\(sa.count)")
        for a in sa { print(attribute("stageInputAttribute", a)) }
    } else { print("stageInputAttributes nil") }
    let fc = f.functionConstantsDictionary
    print("functionConstants count=\(fc.count)")
    for (k, c) in fc.sorted(by: { $0.key < $1.key }) { print("  functionConstant key=\(k) name=\(c.name) type=\(c.type.rawValue) index=\(c.index) required=\(c.required)") }
}
func functions(_ fixture: String) {
    let l = library(fixture)
    for name in l.functionNames.sorted() {
        let f = function(fixture, name)
        properties(f)
        let sel = NSSelectorFromString("reflectionForFunctionWithName:")
        guard let r = (l as AnyObject).perform(sel, with: name)?.takeUnretainedValue() as? NSObject else { print("library nil"); continue }
        let annotation = r.value(forKey: "userAnnotation").map { "\($0)" } ?? "nil"
        let b = (r.value(forKey: "bindings") as? [MTLBinding]) ?? []
        for x in lines("library userAnnotation=\(annotation)", b) { print(x) }
    }
}
func computeLines(_ r: MTLComputePipelineReflection) -> [String] { lines("compute", r.bindings) }
func renderLines(_ r: MTLRenderPipelineReflection) -> [String] {
    lines("vertex", r.vertexBindings) + lines("fragment", r.fragmentBindings) + lines("tile", r.tileBindings)
        + lines("object", r.objectBindings) + lines("mesh", r.meshBindings)
}
// Options 1 in full; options 0, 2 and 3 as `nil`, `same` (as options 1) or in full.
func options(_ make: (MTLPipelineOption) throws -> [String]?) {
    var first: [String]? = nil
    for raw: UInt in [1, 0, 2, 3] {
        do {
            let got = try make(MTLPipelineOption(rawValue: raw))
            if raw == 1 {
                first = got
                print("options=1 \(got == nil ? "nil" : "present")")
                for l in got ?? [] { print(l) }
            } else if got == nil {
                print("options=\(raw) nil")
            } else if got == first {
                print("options=\(raw) same")
            } else {
                print("options=\(raw) differs")
                for l in got ?? [] { print(l) }
            }
        } catch { print("options=\(raw) error \(error)") }
    }
}
func compute(_ make: (MTLPipelineOption, inout MTLAutoreleasedComputePipelineReflection?) throws -> MTLComputePipelineState) {
    options { o in
        var r: MTLAutoreleasedComputePipelineReflection?
        _ = try make(o, &r)
        return r.map(computeLines)
    }
}
func render(_ make: (MTLPipelineOption, inout MTLAutoreleasedRenderPipelineReflection?) throws -> MTLRenderPipelineState) {
    options { o in
        var r: MTLAutoreleasedRenderPipelineReflection?
        _ = try make(o, &r)
        return r.map(renderLines)
    }
}
func constants(_ extra: Bool) -> MTLFunctionConstantValues {
    let c = MTLFunctionConstantValues()
    var e = extra
    var s: Int32 = 3
    c.setConstantValue(&e, type: .bool, index: 0)
    c.setConstantValue(&s, type: .int, index: 1)
    return c
}
func kernel(_ fixture: String, _ name: String, _ c: MTLFunctionConstantValues? = nil) {
    let f = function(fixture, name, c)
    compute { o, r in try device.makeComputePipelineState(function: f, options: o, reflection: &r) }
}
let fixtures = ["argument_buffers", "buffers", "render", "stage_in", "tables", "tessellation", "textures", "tile", "vertex_inputs", "function_kinds", "function_types", "encoders", "encoder_types", "encoder_arrays", "leftovers", "tables_in_arguments", "annotations", "specialized_reads", "mesh_functions", "specialized_inputs", "constant_pointers", "function_addresses", "constant_initializers", "constant_loops", "constant_branches"]
var cases: [(String, () -> Void)] = fixtures.map { f in ("functions_\(f)", { functions(f) }) }
cases += [
    ("buffers_extra_false", { kernel("buffers", "reflection_buffers", constants(false)) }),
    ("buffers_extra_true", { kernel("buffers", "reflection_buffers", constants(true)) }),
    ("order", { kernel("buffers", "reflection_order") }),
    ("threadgroup", { kernel("buffers", "reflection_threadgroup") }),
    ("textures", { kernel("textures", "reflection_textures") }),
    ("argument_buffers", { kernel("argument_buffers", "reflection_argument_buffers") }),
    ("encoders_args", { kernel("encoders", "encoders_args") }),
    ("struct_arrays", { kernel("encoder_arrays", "encoder_arrays") }),
    ("tables", { kernel("tables", "reflection_tables") }),
    ("stage_in", {
        let d = MTLComputePipelineDescriptor()
        d.computeFunction = function("stage_in", "reflection_stage_in")
        let s = MTLStageInputOutputDescriptor()
        s.attributes[0].format = .float4; s.attributes[0].offset = 0; s.attributes[0].bufferIndex = 1
        s.attributes[1].format = .uint; s.attributes[1].offset = 0; s.attributes[1].bufferIndex = 2
        s.layouts[1].stride = 16; s.layouts[1].stepFunction = .threadPositionInGridX
        s.layouts[2].stride = 4; s.layouts[2].stepFunction = .threadPositionInGridX
        d.stageInputDescriptor = s
        compute { o, r in try device.makeComputePipelineState(descriptor: d, options: o, reflection: &r) }
    }),
    ("render_vertex_descriptor", {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = function("render", "reflection_vertex")
        d.fragmentFunction = function("render", "reflection_fragment")
        d.colorAttachments[0].pixelFormat = .rgba8Unorm
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float4; vd.attributes[0].offset = 0; vd.attributes[0].bufferIndex = 0
        vd.attributes[1].format = .float2; vd.attributes[1].offset = 16; vd.attributes[1].bufferIndex = 0
        vd.attributes[2].format = .half4; vd.attributes[2].offset = 0; vd.attributes[2].bufferIndex = 2
        vd.layouts[0].stride = 24
        vd.layouts[2].stride = 8
        d.vertexDescriptor = vd
        render { o, r in try device.makeRenderPipelineState(descriptor: d, options: o, reflection: &r) }
    }),
    ("render_plain", {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = function("render", "reflection_vertex_plain")
        d.fragmentFunction = function("render", "reflection_fragment_solid")
        d.colorAttachments[0].pixelFormat = .rgba8Unorm
        render { o, r in try device.makeRenderPipelineState(descriptor: d, options: o, reflection: &r) }
    }),
    ("tile", {
        let d = MTLTileRenderPipelineDescriptor()
        d.tileFunction = function("tile", "reflection_tile")
        d.colorAttachments[0].pixelFormat = .rgba8Unorm
        d.threadgroupSizeMatchesTileSize = true
        render { o, r in try device.makeRenderPipelineState(tileDescriptor: d, options: o, reflection: &r) }
    }),
]
// The descriptors' buffers (`vertexBuffer.N`, `stageInBuffer.N`): which a descriptor lists, in
// what order, of what size, and when one is used.
func vertexDescriptor(_ attrs: [(Int, MTLVertexFormat, Int, Int)], _ layouts: [(Int, Int)]) -> MTLVertexDescriptor {
    let vd = MTLVertexDescriptor()
    for (i, f, off, b) in attrs { vd.attributes[i].format = f; vd.attributes[i].offset = off; vd.attributes[i].bufferIndex = b }
    for (b, s) in layouts { vd.layouts[b].stride = s }
    return vd
}
func renderOf(_ v: MTLFunction, _ f: MTLFunction, _ vd: MTLVertexDescriptor?, _ setup: (MTLRenderPipelineDescriptor) -> Void = { _ in }) {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = v
    d.fragmentFunction = f
    d.colorAttachments[0].pixelFormat = .rgba8Unorm
    d.vertexDescriptor = vd
    setup(d)
    render { o, r in try device.makeRenderPipelineState(descriptor: d, options: o, reflection: &r) }
}
func stageIn(_ attrs: [(Int, MTLAttributeFormat, Int, Int)], _ layouts: [(Int, Int)]) {
    let d = MTLComputePipelineDescriptor()
    d.computeFunction = function("stage_in", "reflection_stage_in")
    let s = MTLStageInputOutputDescriptor()
    for (i, f, off, b) in attrs { s.attributes[i].format = f; s.attributes[i].offset = off; s.attributes[i].bufferIndex = b }
    for (b, st) in layouts { s.layouts[b].stride = st; s.layouts[b].stepFunction = .threadPositionInGridX }
    d.stageInputDescriptor = s
    compute { o, r in try device.makeComputePipelineState(descriptor: d, options: o, reflection: &r) }
}
func useUV(_ v: Bool) -> MTLFunctionConstantValues {
    let c = MTLFunctionConstantValues()
    var b = v
    c.setConstantValue(&b, type: .bool, index: 0)
    return c
}
let base: [(Int, MTLVertexFormat, Int, Int)] = [(0, .float4, 0, 0), (1, .float2, 16, 0), (2, .half4, 0, 2)]
let two: [(Int, MTLVertexFormat, Int, Int)] = [(0, .float4, 0, 0), (1, .float2, 0, 1)]
let vertex = { function("render", "reflection_vertex") }
let fragment = { function("render", "reflection_fragment") }
cases += [
    ("vertex_undeclared", { renderOf(vertex(), fragment(), vertexDescriptor(base + [(5, .float, 0, 4)], [(0, 24), (2, 8), (4, 4)])) }),
    ("vertex_shared_mixed", { renderOf(vertex(), fragment(), vertexDescriptor(base + [(5, .float, 20, 0)], [(0, 24), (2, 8)])) }),
    ("vertex_buffer_order", { renderOf(vertex(), fragment(), vertexDescriptor([(0, .float4, 0, 6), (1, .float2, 0, 3), (2, .half4, 8, 3)], [(6, 16), (3, 16)])) }),
    // MTLBufferLayoutStrideDynamic (NSUIntegerMax).
    ("vertex_dynamic_stride", { renderOf(vertex(), fragment(), vertexDescriptor(base, [(0, Int(bitPattern: UInt.max)), (2, 8)])) }),
    ("vertex_instance_step", {
        let vd = vertexDescriptor(base, [(0, 24), (2, 8)])
        vd.layouts[2].stepFunction = .perInstance
        renderOf(vertex(), fragment(), vd)
    }),
    ("vertex_unread", { renderOf(function("vertex_inputs", "reflection_unread"), function("vertex_inputs", "reflection_solid"), vertexDescriptor(two, [(0, 16), (1, 8)])) }),
    ("vertex_gated_false", { renderOf(function("vertex_inputs", "reflection_gated", useUV(false)), function("vertex_inputs", "reflection_solid"), vertexDescriptor(two, [(0, 16), (1, 8)])) }),
    ("vertex_gated_true", { renderOf(function("vertex_inputs", "reflection_gated", useUV(true)), function("vertex_inputs", "reflection_solid"), vertexDescriptor(two, [(0, 16), (1, 8)])) }),
    ("vertex_tessellation", {
        let vd = vertexDescriptor([(0, .float4, 0, 0), (3, .float4, 0, 1)], [(0, 16), (1, 16)])
        vd.layouts[0].stepFunction = .perPatchControlPoint
        vd.layouts[1].stepFunction = .perPatch
        renderOf(function("tessellation", "reflection_tessellation"), function("render", "reflection_fragment_solid"), vd) { d in
            d.maxTessellationFactor = 16
            d.tessellationControlPointIndexType = .none
        }
    }),
    ("stage_in_undeclared", { stageIn([(0, .float4, 0, 1), (1, .uint, 0, 2), (4, .float, 0, 5)], [(1, 16), (2, 4), (5, 4)]) }),
    ("stage_in_buffer_order", { stageIn([(0, .float4, 0, 7), (1, .uint, 0, 3)], [(7, 16), (3, 4)]) }),
    // MTLBufferLayoutStrideDynamic (NSUIntegerMax) in a stage input descriptor.
    ("stage_in_dynamic_stride", { stageIn([(0, .float4, 0, 1), (1, .uint, 0, 2)], [(1, Int(bitPattern: UInt.max)), (2, 4)]) }),
]
// The MTLFunction properties of a function made with constant values
// (`newFunctionWithName:constantValues:error:`).
func kindsConstants(gate: Bool, defined: Bool) -> MTLFunctionConstantValues {
    let c = MTLFunctionConstantValues()
    var g = gate
    c.setConstantValue(&g, type: .bool, index: 1)
    if defined {
        var d = true
        c.setConstantValue(&d, type: .bool, index: 0)
    }
    return c
}
func kindsValues() -> MTLFunctionConstantValues {
    let c = MTLFunctionConstantValues()
    var v: Int32 = 5
    c.setConstantValue(&v, type: .int, index: 2)
    return c
}
cases += [
    ("specialized_gated_false", { properties(function("vertex_inputs", "reflection_gated", useUV(false))) }),
    ("specialized_gated_true", { properties(function("vertex_inputs", "reflection_gated", useUV(true))) }),
    ("specialized_constants_gate_false", { properties(function("function_kinds", "kinds_constants", kindsConstants(gate: false, defined: true))) }),
    ("specialized_constants_undefined", { properties(function("function_kinds", "kinds_constants", kindsConstants(gate: true, defined: false))) }),
    ("specialized_values", { properties(function("function_kinds", "kinds_values", kindsValues())) }),
    ("specialized_none", { properties(function("function_kinds", "kinds_none", MTLFunctionConstantValues())) }),
]
// Encoders made from a reflected buffer binding (`newArgumentEncoderWithBufferBinding:`), beside
// the function's own (`newArgumentEncoderWithBufferIndex:`): their encoded length and alignment,
// the offsets of the constants `constantDataAtIndex:` reaches, and the nested encoder of a
// pointer to an argument buffer.
func encoderLines(_ label: String, _ e: MTLArgumentEncoder, _ constants: [Int], _ nested: [Int]) -> [String] {
    guard let buffer = device.makeBuffer(length: max(e.encodedLength, 16) * 2, options: .storageModeShared) else { fail("no buffer") }
    e.setArgumentBuffer(buffer, offset: 0)
    let base = Int(bitPattern: buffer.contents())
    var out = ["\(label) encodedLength=\(e.encodedLength) alignment=\(e.alignment)"]
    for i in constants { out.append("  constant index=\(i) offset=\(Int(bitPattern: e.constantData(at: i)) - base)") }
    for i in nested {
        if let n = e.makeArgumentEncoderForBuffer(atIndex: i) {
            out.append("  nested index=\(i) encodedLength=\(n.encodedLength) alignment=\(n.alignment)")
        } else { out.append("  nested index=\(i) nil") }
    }
    return out
}
func libraryBinding(_ fixture: String, _ function: String, _ name: String) -> MTLBufferBinding {
    let sel = NSSelectorFromString("reflectionForFunctionWithName:")
    guard let r = (library(fixture) as AnyObject).perform(sel, with: function)?.takeUnretainedValue() as? NSObject,
          let bindings = r.value(forKey: "bindings") as? [MTLBinding],
          let b = bindings.first(where: { $0.name == name }) as? MTLBufferBinding else { fail("no library binding \(name)") }
    return b
}
func pipelineBinding(_ fixture: String, _ function: String, _ name: String) -> MTLBufferBinding {
    var r: MTLAutoreleasedComputePipelineReflection?
    do { _ = try device.makeComputePipelineState(function: library(fixture).makeFunction(name: function)!, options: MTLPipelineOption(rawValue: 1), reflection: &r) } catch { fail("pipeline: \(error)") }
    guard let b = r?.bindings.first(where: { $0.name == name }) as? MTLBufferBinding else { fail("no pipeline binding \(name)") }
    return b
}
let encoderArgs = (constants: [7, 8, 20, 21], nested: [2, 12, 15])
let encoderImp = (constants: [3], nested: [0])
func encoderCase(_ bufferIndex: Int, _ name: String, _ shape: (constants: [Int], nested: [Int])) {
    let f = function("argument_buffers", "reflection_argument_buffers")
    for l in encoderLines("function", f.makeArgumentEncoder(bufferIndex: bufferIndex), shape.constants, shape.nested) { print(l) }
    let lib = libraryBinding("argument_buffers", "reflection_argument_buffers", name)
    for l in encoderLines("library", device.makeArgumentEncoder(bufferBinding: lib), shape.constants, shape.nested) { print(l) }
    let pipe = pipelineBinding("argument_buffers", "reflection_argument_buffers", name)
    for l in encoderLines("pipeline", device.makeArgumentEncoder(bufferBinding: pipe), shape.constants, shape.nested) { print(l) }
}
// Where an encoder writes each array element and struct member: every argument of
// argument_buffers' args whose place an array's or struct's stride decides, set to a distinct
// texture, sampler or buffer, then each one's 8-byte resource ID or address found in a zeroed
// argument buffer.
func placements(_ label: String, _ e: MTLArgumentEncoder) -> [String] {
    guard let buffer = device.makeBuffer(length: 512, options: .storageModeShared) else { fail("no buffer") }
    memset(buffer.contents(), 0, 512)
    e.setArgumentBuffer(buffer, offset: 0)
    var keep: [AnyObject] = []
    var values: [(Int, UInt64)] = []
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false)
    for i in [0, 3, 4, 5, 22, 24, 26] {
        guard let t = device.makeTexture(descriptor: td) else { fail("no texture") }
        keep.append(t); e.setTexture(t, index: i); values.append((i, t.gpuResourceID._impl))
    }
    for (k, i) in [1, 16, 17, 23, 25, 27].enumerated() {
        let sd = MTLSamplerDescriptor()
        sd.supportArgumentBuffers = true
        sd.lodMinClamp = Float(k)
        guard let smp = device.makeSamplerState(descriptor: sd) else { fail("no sampler") }
        keep.append(smp); e.setSamplerState(smp, index: i); values.append((i, smp.gpuResourceID._impl))
    }
    for i in [2, 18, 19] {
        guard let b = device.makeBuffer(length: 64) else { fail("no buffer") }
        keep.append(b); e.setBuffer(b, offset: 0, index: i); values.append((i, b.gpuAddress))
    }
    var out = ["\(label) encodedLength=\(e.encodedLength)"]
    let bytes = buffer.contents()
    for (i, v) in values {
        let at = stride(from: 0, to: 512 - 7, by: 4).filter { bytes.loadUnaligned(fromByteOffset: $0, as: UInt64.self) == v }
        out.append("  index=\(i) offsets=\(at)")
    }
    _ = keep
    return out
}
cases += [
    ("encoder_args_places", {
        let f = function("argument_buffers", "reflection_argument_buffers")
        for l in placements("function", f.makeArgumentEncoder(bufferIndex: 0)) { print(l) }
        let lib = libraryBinding("argument_buffers", "reflection_argument_buffers", "args")
        for l in placements("library", device.makeArgumentEncoder(bufferBinding: lib)) { print(l) }
        let pipe = pipelineBinding("argument_buffers", "reflection_argument_buffers", "args")
        for l in placements("pipeline", device.makeArgumentEncoder(bufferBinding: pipe)) { print(l) }
    }),
]
cases += [
    ("encoder_args", { encoderCase(0, "args", encoderArgs) }),
    ("encoder_imp", { encoderCase(1, "imp", encoderImp) }),
    // A plain buffer's binding: what Metal does with it.
    ("encoder_plain_buffer", {
        let b = libraryBinding("argument_buffers", "reflection_argument_buffers", "out")
        let e = device.makeArgumentEncoder(bufferBinding: b)
        print("plain encodedLength=\(e.encodedLength) alignment=\(e.alignment)")
    }),
    // The same through the pipeline's binding, and the function's encoder at the plain buffer's
    // index, each run with and without MTL_DEBUG_LAYER=1 (whether API validation stops them).
    ("encoder_plain_buffer_pipeline", {
        let b = pipelineBinding("argument_buffers", "reflection_argument_buffers", "out")
        let e = device.makeArgumentEncoder(bufferBinding: b)
        print("plain encodedLength=\(e.encodedLength) alignment=\(e.alignment)")
    }),
    // Plain buffers whose pointee is a struct (no argument buffer): the encoder their library
    // binding makes, its length, alignment and constant offsets at indices 0 to 2.
    ("encoder_plain_struct", {
        for name in ["plain", "nested", "counter"] {
            let e = device.makeArgumentEncoder(bufferBinding: libraryBinding("buffers", "reflection_buffers", name))
            print("\(name) encodedLength=\(e.encodedLength) alignment=\(e.alignment)")
            guard e.encodedLength > 0, let buffer = device.makeBuffer(length: e.encodedLength, options: .storageModeShared) else { continue }
            e.setArgumentBuffer(buffer, offset: 0)
            let base = Int(bitPattern: buffer.contents())
            for i in 0..<3 { print("  constant index=\(i) offset=\(Int(bitPattern: e.constantData(at: i)) - base)") }
        }
    }),
    ("encoder_plain_buffer_function", {
        let f = function("argument_buffers", "reflection_argument_buffers")
        let e = f.makeArgumentEncoder(bufferIndex: 2)
        print("plain encodedLength=\(e.encodedLength) alignment=\(e.alignment)")
    }),
]
// Every buffer binding's encoder (`newArgumentEncoderWithBufferBinding:`) of a kernel, through its
// library's and its pipeline's reflection: its length and alignment and, unless `create`, the
// offsets `constantDataAtIndex:` answers at indices 0 to 7; `only` names one binding. Run as
// `probe <dir> enc:<fixture>:<kernel>[:create[:<binding>]]`, outside `list`.
func bindingEncoders(_ fixture: String, _ kernel: String, _ offsets: Bool, _ only: String?) {
    let sel = NSSelectorFromString("reflectionForFunctionWithName:")
    guard let r = (library(fixture) as AnyObject).perform(sel, with: kernel)?.takeUnretainedValue() as? NSObject,
          let lib = r.value(forKey: "bindings") as? [MTLBinding] else { fail("no library reflection \(kernel)") }
    var pr: MTLAutoreleasedComputePipelineReflection?
    do { _ = try device.makeComputePipelineState(function: function(fixture, kernel), options: MTLPipelineOption(rawValue: 1), reflection: &pr) } catch { fail("pipeline: \(error)") }
    for (label, bindings) in [("library", lib), ("pipeline", pr?.bindings ?? [])] {
        for binding in bindings {
            guard let b = binding as? MTLBufferBinding, only == nil || b.name == only else { continue }
            let e = device.makeArgumentEncoder(bufferBinding: b)
            print("\(label) \(b.name) encodedLength=\(e.encodedLength) alignment=\(e.alignment)")
            guard offsets, e.encodedLength > 0, let buffer = device.makeBuffer(length: e.encodedLength, options: .storageModeShared) else { continue }
            e.setArgumentBuffer(buffer, offset: 0)
            let base = Int(bitPattern: buffer.contents())
            print("  constants " + (0..<8).map { "\($0):\(Int(bitPattern: e.constantData(at: $0)) - base)" }.joined(separator: " "))
        }
    }
}
// The bindings no earlier case measured: an explicit imageblock in a tile function and in a compute
// kernel, a fragment function's imageblock data, render pipeline states inside an argument buffer,
// implicit indices after gated arguments, indices a function constant computes, and descriptor
// attributes whose buffer has no layout (each recorded without API validation; their runs under
// MTL_DEBUG_LAYER=1 are recorded in the answers' header).
func leftoversGate(_ gate: Bool) -> MTLFunctionConstantValues {
    let c = MTLFunctionConstantValues()
    var g = gate
    c.setConstantValue(&g, type: .bool, index: 0)
    return c
}
func leftoversBase(_ base: Int32) -> MTLFunctionConstantValues {
    let c = MTLFunctionConstantValues()
    var b = base
    c.setConstantValue(&b, type: .int, index: 1)
    return c
}
cases += [
    ("tile_explicit", {
        let d = MTLTileRenderPipelineDescriptor()
        d.tileFunction = function("leftovers", "reflection_tile_explicit")
        d.colorAttachments[0].pixelFormat = .rgba8Unorm
        d.threadgroupSizeMatchesTileSize = true
        render { o, r in try device.makeRenderPipelineState(tileDescriptor: d, options: o, reflection: &r) }
    }),
    ("kernel_explicit", { kernel("leftovers", "reflection_kernel_explicit") }),
    ("fragment_data", { renderOf(function("leftovers", "reflection_cover"), function("leftovers", "reflection_fragment_data"), nil) }),
    ("pipeline_states", { kernel("leftovers", "reflection_pipeline_states") }),
    ("gated_indices_false", { kernel("leftovers", "reflection_gated_indices", leftoversGate(false)) }),
    ("gated_indices_true", { kernel("leftovers", "reflection_gated_indices", leftoversGate(true)) }),
    ("computed_indices_0", { kernel("leftovers", "reflection_computed_indices", leftoversBase(0)) }),
    ("computed_indices_3", { kernel("leftovers", "reflection_computed_indices", leftoversBase(3)) }),
    // An attribute whose buffer has no layout: one the function reads (attribute 2, buffer 2),
    // one it does not (attribute 1, buffer 1), and a stage input's (attribute 1, buffer 2).
    ("vertex_no_layout", { renderOf(vertex(), fragment(), vertexDescriptor(base, [(0, 24)])) }),
    ("vertex_no_layout_unread", { renderOf(function("vertex_inputs", "reflection_unread"), function("vertex_inputs", "reflection_solid"), vertexDescriptor(two, [(0, 16)])) }),
    ("stage_in_no_layout", { stageIn([(0, .float4, 0, 1), (1, .uint, 0, 2)], [(1, 16)]) }),
]
// Visible function tables inside an argument buffer.
cases += [
    ("tables_in_arguments", { kernel("tables_in_arguments", "reflection_tables_in_arguments") }),
]
// A user annotation.
cases += [
    ("annotated", { kernel("annotations", "reflection_annotated") }),
]
// A vertex input a function constant leaves unread: the attribute and the pipeline's buffer.
func branchRead(_ read: Bool) -> MTLFunction { function("specialized_reads", "reflection_branch_read", useUV(read)) }
cases += [
    ("specialized_branch_false", { properties(branchRead(false)) }),
    ("specialized_branch_true", { properties(branchRead(true)) }),
    ("render_branch_false", { renderOf(branchRead(false), function("specialized_reads", "reflection_branch_solid"), vertexDescriptor(two, [(0, 16), (1, 8)])) }),
    ("render_branch_true", { renderOf(branchRead(true), function("specialized_reads", "reflection_branch_solid"), vertexDescriptor(two, [(0, 16), (1, 8)])) }),
]
// A mesh render pipeline with and without an object function.
func meshPipeline(_ object: Bool) {
    let d = MTLMeshRenderPipelineDescriptor()
    if object { d.objectFunction = function("mesh_functions", "reflection_object") }
    d.meshFunction = function("mesh_functions", "reflection_mesh")
    d.fragmentFunction = function("mesh_functions", "reflection_mesh_fragment")
    d.colorAttachments[0].pixelFormat = .rgba8Unorm
    options { o in
        let (_, r) = try device.makeRenderPipelineState(descriptor: d, options: o)
        return r.map(renderLines)
    }
}
cases += [
    ("mesh_pipeline", { meshPipeline(true) }),
    ("mesh_pipeline_no_object", { meshPipeline(false) }),
]
// A kernel's stage inputs, a post-tessellation function's control-point and per-patch inputs, and
// argument bindings of each kind that a function constant leaves unread: the attributes and the
// pipelines' bindings.
func gatedKernel(_ read: Bool) -> MTLFunction { function("specialized_inputs", "reflection_gated_stage_in", useUV(read)) }
func gatedTessellation(_ read: Bool) -> MTLFunction { function("specialized_inputs", "reflection_gated_tessellation", useUV(read)) }
func gatedStageIn(_ read: Bool) {
    let d = MTLComputePipelineDescriptor()
    d.computeFunction = gatedKernel(read)
    let s = MTLStageInputOutputDescriptor()
    s.attributes[0].format = .float4; s.attributes[0].offset = 0; s.attributes[0].bufferIndex = 1
    s.attributes[1].format = .uint; s.attributes[1].offset = 0; s.attributes[1].bufferIndex = 2
    s.layouts[1].stride = 16; s.layouts[1].stepFunction = .threadPositionInGridX
    s.layouts[2].stride = 4; s.layouts[2].stepFunction = .threadPositionInGridX
    d.stageInputDescriptor = s
    compute { o, r in try device.makeComputePipelineState(descriptor: d, options: o, reflection: &r) }
}
func gatedTessellationPipeline(_ read: Bool) {
    let vd = vertexDescriptor([(0, .float4, 0, 0), (1, .float4, 16, 0), (3, .float4, 0, 1), (4, .float4, 16, 1)], [(0, 32), (1, 32)])
    vd.layouts[0].stepFunction = .perPatchControlPoint
    vd.layouts[1].stepFunction = .perPatch
    renderOf(gatedTessellation(read), function("specialized_inputs", "reflection_gated_solid"), vd) { d in
        d.maxTessellationFactor = 16
        d.tessellationControlPointIndexType = .none
    }
}
cases += [
    ("specialized_gated_kernel_false", { properties(gatedKernel(false)) }),
    ("specialized_gated_kernel_true", { properties(gatedKernel(true)) }),
    ("specialized_gated_tessellation_false", { properties(gatedTessellation(false)) }),
    ("specialized_gated_tessellation_true", { properties(gatedTessellation(true)) }),
    ("stage_in_gated_false", { gatedStageIn(false) }),
    ("stage_in_gated_true", { gatedStageIn(true) }),
    ("tessellation_gated_false", { gatedTessellationPipeline(false) }),
    ("tessellation_gated_true", { gatedTessellationPipeline(true) }),
    ("specialized_gated_control_points_false", { properties(function("specialized_inputs", "reflection_gated_control_points", useUV(false))) }),
    ("specialized_gated_control_points_true", { properties(function("specialized_inputs", "reflection_gated_control_points", useUV(true))) }),
    ("resources_gated_false", { kernel("specialized_inputs", "reflection_gated_resources", useUV(false)) }),
    ("resources_gated_true", { kernel("specialized_inputs", "reflection_gated_resources", useUV(true)) }),
]
if args[2] == "list" {
    for (k, _) in cases { print(k) }
    exit(0)
}
if args[2].hasPrefix("enc:") {
    let p = args[2].split(separator: ":", omittingEmptySubsequences: false).map(String.init)
    guard p.count >= 3 else { fail("enc:<fixture>:<kernel>[:create[:<binding>]]") }
    bindingEncoders(p[1], p[2], !(p.count > 3 && p[3] == "create"), p.count > 4 ? p[4] : nil)
    exit(0)
}
guard let run = cases.first(where: { $0.0 == args[2] })?.1 else { fail("no case \(args[2])") }
run()
