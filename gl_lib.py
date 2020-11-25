import ctypes as ct
import os
from typing import Union
import struct
import zlib
import math

from pyglet.gl import *


class Vector:
    def __init__(self, x, y, z=0., w=1.):
        self.x, self.y, self.z, self.w = x, y, z, w

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other, self.w * other)

    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other, self.w / other)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def length(self):
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** .5

    def normalised(self):
        return self / self.length()

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __iter__(self):
        return iter((self.x, self.y, self.z, self.w))

    def tuple(self, size=4):
        return (self.x, self.y, self.z, self.w)[:size]

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


class Matrix:
    def __init__(self, data=None):
        self.m = data if data else Matrix.identity()

    @staticmethod
    def identity():
        return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    def mult_mat(self, other):  # self = other * self
        new = []
        a = other.m
        b = self.m

        for x in range(0, 16, 4):
            for y in range(0, 4):
                new.append(
                    a[y + 0] * b[x + 0] +
                    a[y + 4] * b[x + 1] +
                    a[y + 8] * b[x + 2] +
                    a[y + 12] * b[x + 3]
                )

        self.m = new
        return self

    def __mul__(self, other):
        new = []
        a = self.m
        b = other.m

        for x in range(0, 16, 4):
            for y in range(0, 4):
                new.append(
                    a[y + 0] * b[x + 0] +
                    a[y + 4] * b[x + 1] +
                    a[y + 8] * b[x + 2] +
                    a[y + 12] * b[x + 3]
                )

        return Matrix(new)

    def mult_vec(self, vec):
        out = []
        for i in range(4):
            out.append(
                self.m[i] * vec.x +
                self.m[i + 4] * vec.y +
                self.m[i + 8] * vec.z +
                self.m[i + 12] * vec.w
            )
        return Vector(*out)

    def perspective(self, fovy, aspect, near, far):
        f = 1 / math.tan(math.radians(fovy) / 2)
        return self.mult_mat(Matrix([f / aspect, 0, 0, 0,
                                     0, f, 0, 0,
                                     0, 0, (near + far) / (near - far), -1,
                                     0, 0, 2 * (near * far) / (near - far), 0]))

    def translate(self, pos):
        return self.mult_mat(Matrix([1, 0, 0, 0,
                                     0, 1, 0, 0,
                                     0, 0, 1, 0,
                                     pos.x, pos.y, pos.z, 1]))

    def rotate_x(self, angle):
        angle = math.radians(angle)
        return self.mult_mat(Matrix([1, 0, 0, 0,
                                     0, math.cos(angle), math.sin(angle), 0,
                                     0, -math.sin(angle), math.cos(angle), 0,
                                     0, 0, 0, 1]))

    def rotate_y(self, angle):
        angle = math.radians(angle)
        return self.mult_mat(Matrix([math.cos(angle), 0, -math.sin(angle), 0,
                                     0, 1, 0, 0,
                                     math.sin(angle), 0, math.cos(angle), 0,
                                     0, 0, 0, 1]))

    def rotate_z(self, angle):
        angle = math.radians(angle)
        return self.mult_mat(Matrix([math.cos(angle), math.sin(angle), 0, 0,
                                     -math.sin(angle), math.cos(angle), 0, 0,
                                     0, 0, 1, 0,
                                     0, 0, 0, 1]))

    def scale(self, factor):
        return self.mult_mat(Matrix([factor, 0, 0, 0,
                                     0, factor, 0, 0,
                                     0, 0, factor, 0,
                                     0, 0, 0, 1]))

    def reset(self):
        self.m = Matrix.identity()


class Uniform:
    TYPES_FUNC = {
        'mat4': lambda self, v: glUniformMatrix4fv(self.id, 1, GL_FALSE, (ct.c_float * 16)(*v)),
        'mat2': lambda self, v: glUniformMatrix2fv(self.id, 1, GL_FALSE, (ct.c_float * 4)(*v)),
        'int': lambda self, v: glUniform1i(self.id, v),
        'float': lambda self, v: glUniform1f(self.id, v),
        'vec2': lambda self, v: glUniform2f(self.id, *v),
        'vec3': lambda self, v: glUniform3f(self.id, *v),
        'vec4': lambda self, v: glUniform4f(self.id, *v),
        'sampler2D': lambda self, v: glUniform1i(self.id, v),
    }

    def __init__(self, name, identif, typ):
        self.name = name
        self.id = identif
        self.funcs = Uniform.TYPES_FUNC[typ]

    @classmethod
    def from_prgm(cls, prgm, name, typ):
        i = glGetUniformLocation(prgm, name.encode('latin'))
        return cls(name, i, typ)

    def __call__(self, val):
        self.funcs(self, val)


class Shader:
    def __init__(self, *args):
        self.uniforms = dict()
        self.program = 0
        if len(args) == 1:
            self.name = args[0]
            self.vert_data = open(args[0] + '.vert', 'r').read()
            self.frag_data = open(args[0] + '.frag', 'r').read()
        elif len(args) == 3:
            self.name, vert_data, frag_data = args
        else:
            raise RuntimeError('invalid arguments')

    @staticmethod
    def gen_pp_str(data):
        return ct.cast(ct.pointer((ct.c_char_p * 1)(data.encode('latin'))), ct.POINTER(ct.POINTER(ct.c_char)))

    def generate(self):
        if self.program != 0:
            return print(self.program)
        status_int = ct.c_int(0)

        vert_ptr = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vert_ptr, 1, Shader.gen_pp_str(self.vert_data), None)
        glCompileShader(vert_ptr)
        glGetShaderiv(vert_ptr, GL_COMPILE_STATUS, ct.byref(status_int))
        if not status_int.value:
            glGetShaderiv(vert_ptr, GL_INFO_LOG_LENGTH, ct.byref(status_int))
            buffer = ct.create_string_buffer(status_int.value)
            glGetShaderInfoLog(vert_ptr, status_int, None, buffer)
            print(buffer.value.decode('latin'))
            raise RuntimeError('vert compile error')

        frag_ptr = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(frag_ptr, 1, Shader.gen_pp_str(self.frag_data), None)
        glCompileShader(frag_ptr)
        glGetShaderiv(frag_ptr, GL_COMPILE_STATUS, ct.byref(status_int))
        if not status_int.value:
            glGetShaderiv(frag_ptr, GL_INFO_LOG_LENGTH, ct.byref(status_int))
            buffer = ct.create_string_buffer(status_int.value)
            glGetShaderInfoLog(frag_ptr, status_int, None, buffer)
            print(buffer.value.decode('latin'))
            raise RuntimeError('frag compile error')

        self.program = glCreateProgram()
        glAttachShader(self.program, vert_ptr)
        glAttachShader(self.program, frag_ptr)
        glLinkProgram(self.program)

        glGetProgramiv(self.program, GL_LINK_STATUS, ct.byref(status_int))
        if not status_int.value:
            glDeleteProgram(self.program)
            glDeleteShader(vert_ptr)
            glDeleteShader(frag_ptr)
            raise RuntimeError('prgm linking error')
        glDeleteShader(vert_ptr)
        glDeleteShader(frag_ptr)
        self._search_uniforms(self.vert_data)
        self._search_uniforms(self.frag_data)

    def use(self):
        glUseProgram(self.program)

    def _search_uniforms(self, data):
        for i in data.split('\n'):
            if i.strip().startswith('uniform'):
                name, typ = i.split(' ')[2].strip(';'), i.split(' ')[1]
                self.uniforms[name] = Uniform.from_prgm(self.program, name, typ)

    def uniform_mat(self, uniform, mat):
        glUniformMatrix4fv(self.uniforms[uniform], 1, GL_FALSE, mat.m)

    def uniform_int(self, uniform, val):
        glUniform1i(self.uniforms[uniform], val)

    def uniform_float(self, uniform, val):
        glUniform1f(self.uniforms[uniform], val)

    def uniform_vec(self, uniform, val):
        glUniform3f(self.uniforms[uniform], *val)

    def __getitem__(self, item):
        if item in self.uniforms:
            return self.uniforms[item]
        print('Invalid Uniform:  ', item)
        self.uniforms[item] = lambda *a: None
        return self.uniforms[item]


class VertexData:
    def __init__(self, data):
        self._data = data
        self.vbo, self.vao = -1, -1

    def generate(self):
        if self.vbo >= 0:
            return

        vbo = ct.pointer(ct.c_uint(0))
        glGenBuffers(1, vbo)
        self.vbo = vbo.contents.value

        vao = ct.pointer(ct.c_uint(0))
        glGenVertexArrays(1, vao)
        self.vao = vao.contents.value

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(self._data), (ct.c_float * len(self._data))(*self._data), GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ct.c_void_p(0))
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * 4, ct.c_void_p(3 * 4))
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * 4, ct.c_void_p(5 * 4))

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)

    def delete(self):
        pass

    def draw(self):
        if self.vbo < 0:
            self.generate()
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, len(self._data) // 8)


class Texture:
    TEXTURE_UNITS = [GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3]

    def __init__(self, auto_dim=None):
        print('new tex')
        self.tex_id = -1
        self.dim = None
        if auto_dim:
            self.generate_from_dim(auto_dim)

    def bind(self, shader: Shader = None, tex_unit=0):
        glActiveTexture(self.TEXTURE_UNITS[tex_unit])
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        if shader:
            shader['MainTexture'](0)

    def generate_from_file(self, name):
        p = ct.pointer(ct.c_uint(0))
        glGenTextures(1, p)
        self.tex_id = p.contents.value

        fs = open(name, 'rb')
        self.dim = struct.unpack('2I', fs.read(8))
        data = zlib.decompress(fs.read())

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *self.dim, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

    def generate_from_dim(self, dim):
        self.dim = dim

        p = ct.pointer(ct.c_uint(0))
        glGenTextures(1, p)
        self.tex_id = p.contents.value

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *dim, 0, GL_RGBA, GL_UNSIGNED_BYTE, ct.c_void_p())

    def __bool__(self):
        return self.tex_id >= 0


class DepthTex:
    def __init__(self, auto_dim=None):
        self.tex_id = -1
        if auto_dim:
            self.generate_from_dim(auto_dim)

    def generate_from_dim(self, dim):
        p = ct.pointer(ct.c_uint(0))
        glGenTextures(1, p)
        self.tex_id = p.contents.value

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, *dim, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, ct.c_void_p())

    def __bool__(self):
        return self.tex_id >= 0


class Model:
    def __init__(self):
        self.model_mat = Matrix()
        self.vertex_data: Union[VertexData, None] = None
        self._vertex_datas = {}
        self.shader: Union[Shader, None] = None
        self._shaders = {}
        self.texture = Texture()

    def load_shader(self, shader_name, force_override=True):
        if force_override or shader_name not in self._shaders:
            self._shaders[shader_name] = Shader(shader_name)
        self.shader = self._shaders[shader_name]
        self.shader.generate()
        return self.shader

    @classmethod
    def _load_model_o(cls, fn):
        fs = open(fn, 'r')
        header = fs.readline().strip().split(' ')
        if header[0] != 'v3t2n3':
            raise RuntimeError('Vertex format not supported')
        vertex_count = [int(i) for i in header[1].split('x')]
        if vertex_count[0] != 3:
            raise RuntimeError('Faces must be tris')

        data = []
        for f in range(vertex_count[1]):
            v_data = []
            for v in range(vertex_count[0]):
                while not (line := fs.readline().strip()):
                    pass
                v_data.append([float(i) for i in line.split(' ')])
            for i in v_data:
                data.extend(i)

        return VertexData(data)

    def load_model(self, fn, force_override=True):
        if not force_override and fn in self._vertex_datas:
            self.vertex_data = self._vertex_datas[fn]
            return self.vertex_data
        ext = os.path.splitext(fn)[1]
        if ext == '.o':
            self._vertex_datas[fn] = self._load_model_o(fn)
        self.vertex_data = self._vertex_datas[fn]
        return self.vertex_data

    def load_texture(self, name):
        return self.texture.generate_from_file(name)

    def draw(self, view_mat, tex_override=None):
        if self.shader:
            self.shader.use()
            self.shader['viewMat'](view_mat.m)
            self.shader['objMat'](self.model_mat.m)
            if tex_override:
                tex_override.bind(self.shader)
            elif self.texture:
                self.texture.bind(self.shader)
        if self.vertex_data:
            self.vertex_data.draw()
        glUseProgram(0)


class RenderImage:
    def __init__(self, dim):
        self.fbo_id = -1
        self.tex: Union[Texture, None] = None
        self.depth: Union[DepthTex, None] = None
        self.dim = dim

    def initiate(self, tex=None):
        self.tex = Texture(self.dim) if tex is None else tex
        self.depth = DepthTex(self.dim)

        p = ct.pointer(ct.c_uint(0))
        glGenFramebuffers(1, p)
        self.fbo_id = p.contents.value

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex.tex_id, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth.tex_id, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def start_render(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

    def finish_render(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # if 'PIL' in globals() and self.dim[0] == 2048:
        #     buffer = (ct.c_ubyte * (self.dim[0] * self.dim[1] * 4))()
        #     self.tex.bind(tex_unit=1)
        #     glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer)
        #     # noinspection PyUnresolvedReferences
        #     img = PIL.Image.frombytes('RGBA', self.dim, bytes(buffer), 'raw').transpose(PIL.Image.FLIP_TOP_BOTTOM)
        #     img.save('image.png')


__all__ = ['Vector', 'Matrix', 'Uniform', 'Shader', 'VertexData', 'Texture', 'Model', 'RenderImage']
