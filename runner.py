import struct
import platform

import wgpu.utils
from wgpu import TextureFormat
from wgpu.backends.wgpu_native import _api as api

MAX_INFO = 1000

MAC_OS = platform.system() == "Darwin"

"""
The fundamental information about any of the many draw commands is the
<vertex_instance, instance_index> pair that is passed to the vertex shader. By using
point-list topology, each call to the vertex shader turns into a single call to the
fragment shader, where the pair is recorded.

(To modify a buffer in the vertex shader requires the feature vertex-writable-storage)

We call various combinations of draw functions and verify that they generate precisely
the pairs (those possibly in a different order) that we expect.
"""
SHADER_SOURCE = (
    f"""
    const MAX_INFO: u32 = {MAX_INFO}u;
    """
    """
    @group(0) @binding(0) var<storage, read_write> data: array<vec2u>;
    @group(0) @binding(1) var<storage, read_write> counter: atomic<u32>;

    struct VertexOutput {
        @builtin(position) position: vec4f,
    }

    const POSITION: vec4f = vec4f(0, 0, 0, 1);

    @vertex
    fn vertex(
        @builtin(vertex_index) vertexIndex: u32,
        @builtin(instance_index) instanceIndex: u32
    ) -> @builtin(position) vec4f {
        let info = vec2u(vertexIndex, instanceIndex);
        let index = atomicAdd(&counter, 1u);
        data[index % MAX_INFO] = info;
        return POSITION;
    }

    @fragment
    fn fragment() -> @location(0) vec4f {
        return vec4f();
    }
"""
)


class Runner:
    REQUIRED_FEATURES = [
        "indirect-first-instance",  # first-instance can be non-zero
        "vertex-writable-storage",  # write to a buffer in vertex shader
        "multi-draw-indirect",  # can call multiDrawIndirect
        *(["multi-draw-indirect-count"] if not MAC_OS else []),
    ]

    def __init__(self):
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        self.device = adapter.request_device(required_features=self.REQUIRED_FEATURES)

        self.output_texture = self.device.create_texture(
            # Actual size is immaterial.  Could just be 1x1
            size=[128, 128],
            format=TextureFormat.rgba8unorm,
            usage="RENDER_ATTACHMENT|COPY_SRC",
        )
        shader = self.device.create_shader_module(code=SHADER_SOURCE)
        self.pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": shader,
            },
            fragment={
                "module": shader,
                "targets": [{"format": self.output_texture.format}],
            },
            primitive={
                "topology": "point-list",
            },
        )

        self.data_buffer = self.device.create_buffer(
            size=MAX_INFO * 2 * 4, usage="STORAGE|COPY_SRC"
        )
        self.counter_buffer = self.device.create_buffer(
            size=4, usage="STORAGE|COPY_SRC|COPY_DST"
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.data_buffer}},
                {"binding": 1, "resource": {"buffer": self.counter_buffer}},
            ],
        )
        self.render_pass_descriptor = {
            "color_attachments": [
                {
                    "clear_value": (0, 0, 0, 0),  # only first value matters
                    "load_op": "clear",
                    "store_op": "store",
                    "view": self.output_texture.create_view(),
                }
            ],
        }

    def run_code(self, data, offset, count_data, count_buffer_offset, max_count):
        # create_buffer_with_data is a convenience function that creates the buffer
        # "mapped_at_creation", copies the data to it, and then unmaps it.
        buffer = self.device.create_buffer_with_data(
            data=struct.pack("i" * len(data), *data), usage="INDIRECT"
        )
        count_buffer = self.device.create_buffer_with_data(
            data=struct.pack("i" * len(data), *data), usage="INDIRECT"
        )

        results = []
        for i in range(2):
            encoder = self.device.create_command_encoder()
            encoder.clear_buffer(self.counter_buffer)
            this_pass = encoder.begin_render_pass(**self.render_pass_descriptor)
            this_pass.set_pipeline(self.pipeline)
            this_pass.set_bind_group(0, self.bind_group)
            if i == 0:
                api.libf.wgpuRenderPassEncoderMultiDrawIndirect(
                    this_pass._internal, buffer._internal, int(offset), int(max_count)
                )
            elif platform.system() != "Darwin":
                api.libf.wgpuRenderPassEncoderMultiDrawIndirectCount(
                    this_pass._internal,
                    buffer._internal,
                    int(offset),
                    count_buffer._internal,
                    int(count_buffer_offset),
                    int(max_count),
                )

            this_pass.end()
            self.device.queue.submit([encoder.finish()])
            results.append(self.get_result())
        print("-------------------------")
        print(
            f"{data=}, {offset=}, {count_data=}, {count_buffer_offset=}, {max_count=}"
        )
        print("DrawIndirect:     ", results[0])
        print("DrawIndirectCount:", results[1])

    def get_result(self):
        """
        Reads the data and count from the GPU and converts it to a sorted list of
        [vertex, instance] pairs.
        """
        count = self.device.queue.read_buffer(self.counter_buffer).cast("i")[0]
        assert count <= MAX_INFO
        if count == 0:
            return []
        info_view = self.device.queue.read_buffer(self.data_buffer, size=count * 2 * 4)
        info = info_view.cast("I", (count, 2)).tolist()
        return sorted(info)


def main():
    runner = Runner()

    runner.run_code([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], 8, [1], 0, 1)
    runner.run_code([1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 8, [0], 0, 1)
    runner.run_code([2, 0, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6], 8, [0], 0, 2)
    runner.run_code([2, 0, 1, 2, 3, 4, 0, 2, 3, 4, 5, 0, 3, 4, 5, 6, 0], 8, [0], 0, 2)

    runner.run_code([3, 0, 2, 2, 1, 2, 0, 2, 2, 3, 4, 0, 2, 2, 5, 6, 0], 8, [0], 0, 3)
    runner.run_code([3, 0, 2, 2, 1, 2, 0, 2, 2, 3, 4, 0, 2, 2, 5, 6, 0], 8, [0], 0, 2)
    runner.run_code([2, 0, 2, 2, 1, 2, 0, 2, 2, 3, 4, 0, 2, 2, 5, 6, 0], 8, [0], 0, 3)

    runner.run_code([2, 0, 2, 2, 1, 2, 0, 2, 2, 3], 8, [0], 0, 2)


if __name__ == "__main__":
    main()
