import PIL.Image
import PIL.PngImagePlugin
import zlib
import os
import struct
import glob


def conv_image(fn):
    img = PIL.Image.open(fn)
    img: PIL.PngImagePlugin.PngImageFile

    def img_iter(img_data):
        for iii in img_data:
            for jjj in iii:
                yield jjj

    data = bytes(img_iter(img.convert('RGBA').getdata()))

    new_fn = os.path.splitext(fn)[0] + '.img'
    with open(new_fn, 'wb') as fs:
        fs.write(struct.pack('2I', img.width, img.height))
        fs.write(zlib.compress(data))


def conv_object(fn):
    vert = []
    tex = []
    norm = []
    face = []

    fs = open(fn, 'r')
    while line := fs.readline():
        inst, line = line.strip().split(' ', 1)
        if inst == 'f':
            face.append([tuple(int(i) - 1 for i in i.split('/')) for i in line.split(' ')])
        elif inst == 'v':
            vert.append(line)
        elif inst == 'vt':
            tex.append(line)
        elif inst == 'vn':
            norm.append(line)
    fs.close()

    data = []
    for f in face:
        if len(f) != 3:
            raise RuntimeError('Faces must be tris')
        for i in f:
            data.append(' '.join((vert[i[0]], tex[i[1]], norm[i[2]])))
    with open(os.path.splitext(fn)[0] + '.o', 'w') as fs:
        fs.write('v3t2n3 3x{}\n'.format(len(data) // 3))
        for i in data:
            fs.write(i + '\n')


def main():
    for i in glob.glob('*.png'):
        conv_image(i)
    for i in glob.glob('*.obj'):
        conv_object(i)


if __name__ == '__main__':
    main()
