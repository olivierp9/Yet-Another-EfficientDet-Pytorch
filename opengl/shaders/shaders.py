class Shaders:
    @staticmethod
    def vertex_src() -> str:
        return """
        # version 330
        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec2 a_texture;
        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        out vec3 v_color;
        out vec2 v_texture;
        void main()
        {
            gl_Position = projection * view * model * vec4(a_position, 1.0);
            v_texture = a_texture;
        }
        """

    @staticmethod
    def fragment_src() -> str:
        return """
        # version 330
        in vec2 v_texture;
        out vec4 out_color;
        float near = 0.1;
        float far = 2;
        
        float LinearizeDepth(float depth) 
        {
            return near * far / (far + depth * (near - far));
            //return depth;
            //float z = depth * 2.0 - 1.0; // back to NDC 
            //return (2.0 * near * far) / (far + near - z * (far - near));	
        }
        
        uniform sampler2D s_texture;
        void main()
        {
            out_color = vec4(vec3(LinearizeDepth(gl_FragCoord.z)), 1.0);
        }
        """


