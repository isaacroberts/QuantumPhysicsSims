import numpy as np
import numba
import vispy as vp


def hsv_to_rgb(hsv):
    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = np.int32(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)

    v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

    #copy alpha
    if hsv.shape[2]>3:
        rgb[3] = hsv[3]

    rgb = np.clip(rgb.astype(float), 0, 1)
    return rgb.reshape(input_shape)

def hsl_to_rgb(hsl):
    rgb = np.zeros(hsl.shape, dtype=np.float)

    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    fr, fg, fb = rgb[...,  0], rgb[..., 1], rgb[..., 2]

    q = np.zeros(l.shape, dtype=np.float)

    lbot = l < 0.5
    q[lbot] = l[lbot] * (1 + s[lbot])

    ltop = lbot == False
    l_ltop, s_ltop = l[ltop], s[ltop]
    q[ltop] = (l_ltop + s_ltop) - (l_ltop * s_ltop)

    p = 2 * l - q
    q_sub_p = q - p

    is_s_zero = s == 0
    l_is_s_zero = l[is_s_zero]
    per_3 = 1./3
    per_6 = 1./6
    two_per_3 = 2./3

    def calc_channel(channel, t):

        t[t < 0] += 1
        t[t > 1] -= 1
        t_lt_per_6 = t < per_6
        t_lt_half = (t_lt_per_6 == False) * (t < 0.5)
        t_lt_two_per_3 = (t_lt_half == False) * (t < two_per_3)
        t_mul_6 = t * 6

        channel[:] = p.copy()
        channel[t_lt_two_per_3] = p[t_lt_two_per_3] + q_sub_p[t_lt_two_per_3] * (4 - t_mul_6[t_lt_two_per_3])
        channel[t_lt_half] = q[t_lt_half].copy()
        channel[t_lt_per_6] = p[t_lt_per_6] + q_sub_p[t_lt_per_6] * t_mul_6[t_lt_per_6]
        channel[is_s_zero] = l_is_s_zero.copy()

    calc_channel(fr, h + per_3)
    calc_channel(fg, h.copy())
    calc_channel(fb, h - per_3)

    #copy alpha
    if hsl.shape[-1]>3:
        rgb[3] = hsl[3]

    rgb = np.clip(rgb, 0, 1)
    return rgb

def abs_ang_to_hsv(abs, ang, hsv=None, mode='black'):
    if hsv is None:
        shp = list(ang.shape)
        if mode=='trans':
            shp.append(4)
        else:
            shp.append(3)
        hsv = np.ones(shp, float)

    hsv[..., 0] = (ang/(2*np.pi))

    if mode=='max':
        hsv[..., 1]=1
        hsv[..., 2]=.5
    elif mode=='black':
        hsv[..., 1] = np.clip(2*abs, .1, 1)
        hsv[..., 2] = np.clip(abs/2, 0.05, 1)
    elif mode=='white':
        hsv[..., 1] = np.clip(2*abs, 0, 1)
        hsv[..., 2] = np.clip(1-abs/2, .1, 1)
    elif mode=='line':
        hsv[..., 1] = np.clip(10*abs, 0, 1)
        hsv[..., 2] = np.clip(.25+abs*.75/2, 0, 1)
    elif mode=='line0cap':
        hsv[..., 1] = np.clip(4*abs, .5, 1)
        hsv[..., 2] = np.clip(abs, .3, 1)
    elif mode=='trans':
        hsv[..., 3] = np.clip(abs/2, 0, 1)

    return hsv

def complex_to_hsv(cpl, hsv=None, mode='black'):
    if mode=='max':
        return abs_ang_to_rgb(1, np.angle(cpl), hsv, mode)
    else:
        return abs_ang_to_rgb(np.abs(cpl), np.angle(cpl), hsv, mode)


def abs_ang_to_rgb(abs, ang, hsv=None, mode='black'):
    return hsl_to_rgb(abs_ang_to_hsv(abs, ang, hsv, mode))

def complex_to_rgb(cpl, hsv=None, mode='black'):
    if mode=='max':
        return abs_ang_to_rgb(1, np.angle(cpl), hsv, mode)
    else:
        return abs_ang_to_rgb(np.abs(cpl), np.angle(cpl), hsv, mode)


class GLSL_Oranges(vp.color.BaseColormap):
    """Transparent orange."""

    glsl_map = """
    vec4 translucent_fire(float t) {
        float alpha;
        /*if (t < .4) {
            alpha = 0.;
        }
        else if (t >= .4 && t < .5) {
            alpha = .1;
        }
        else {
            alpha = 1.;
        }*/
        alpha = t*t;
        return vec4(t, t/2, 0, alpha);
    }
    """

class GLSL_DualColor(vp.color.BaseColormap):
    """ Decodes Hue and Alpha from single number.
        Corresponding encode function is in Voxel3D code


    aLevels = 5
    abs = (abs*aLevels).astype(int)
    c = (abs + ang) / (aLevels+1)
    """

    glsl_map = """
    vec4 translucent_fire(float t) {
        int aLevels = 4;

        t *= (aLevels+1);
        int A = int(t);
        if (A<=0)
            return vec4(0.,0.,0., 0.);
        float a = float(A)/aLevels;
        //a = max(a, 1.f);

        float h = ((t - float(A)));
        h = clamp(h, 0., 1.);
        h *= 2;
        if (h>1)
            h= 2-h;
        float mn = 117;
        float mx = 245;
        float H = (h*(mx-mn)+mn)/360.;


        float s = 1.;
        s = clamp(s, 0, 1);
        float v = 1;
        // float base=1;
        //float v = base+(1-base)*pow(h*2-1, 2);

        //HSV to RGBA - Thanks cargo coding!
        vec3 c = vec3(H, s, v);
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        vec3 o = c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        return vec4(o, a);
    }
    """



class GLSL_HSVColor(vp.color.BaseColormap):
    """ Decodes Hue and Alpha from single number.
        Corresponding encode function is in Voxel3D code

    aLevels = 5
    abs = (abs*aLevels).astype(int)
    c = (abs + ang) / (aLevels+1)
    """

    glsl_map = """
    vec4 translucent_fire(float t) {
        int aLevels = 4;

        t *= (aLevels+1);
        int A = int(t);
        if (A<=0)
            return vec4(0.,0.,0., 0.);
        float a = float(A)/aLevels;
        //a = max(a, 1.f);

        float h = ((t - float(A)));
        h = clamp(h, 0., 1.);
        float H = h;


        float s = 1.;
        float v = 1;

        //HSV to RGBA - Thanks cargo coding!
        vec3 c = vec3(H, s, v);
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        vec3 o = c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        return vec4(o, a);
    }
    """
