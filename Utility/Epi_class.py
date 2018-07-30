from GeometryUtility import *
from random import shuffle

class Camera(object):
    def __init__(self):
        self.C = np.zeros((3, 1))
        self.R = np.identity(3)
        self.K = np.identity(3)
        self.frame = -1
        self.cropping_para = []
        self.image = []
        self.heatmap = []

class Batch(object):
    def __init__(self, nBatch, input_size, heatmap_size, nJoints):
        self.H1 = np.zeros((nBatch, 3, 3))
        self.H2 = np.zeros((nBatch, 3, 3))

        self.H01 = np.zeros((nBatch, 3, 3))
        self.H02 = np.zeros((nBatch, 3, 3))

        self.a1 = np.zeros((nBatch, 1))
        self.b1 = np.zeros((nBatch, 1))
        self.a2 = np.zeros((nBatch, 1))
        self.b2 = np.zeros((nBatch, 1))

        self.image_ref = np.zeros((nBatch, input_size, input_size, 3))
        self.image_src1 = np.zeros((nBatch, input_size, input_size, 3))
        self.image_src2 = np.zeros((nBatch, input_size, input_size, 3))

        self.prob_ref = np.zeros((nBatch, heatmap_size, heatmap_size, nJoints + 1))
        self.prob_src1 = np.zeros((nBatch, heatmap_size, heatmap_size, nJoints + 1))
        self.prob_src2 = np.zeros((nBatch, heatmap_size, heatmap_size, nJoints + 1))


class UnlabeledData(object):
    def __init__(self, nBatch, input_size, heatmap_size, nJoints):
        self.nBatch = nBatch
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.nJoints =  nJoints

    def SetData(self, vET):
        shuffle(vET)
        self.vBatch = []

        n = int(float(len(vET)) / self.nBatch)
        for iBatch in range(n):
            batch = Batch(self.nBatch, self.input_size, self.heatmap_size, self.nJoints)
            for iET in range(self.nBatch):
                batch.image_ref[iET, :, :, :] = vET[iET+iBatch*self.nBatch].camera_ref.image
                batch.image_src1[iET, :, :, :] = vET[iET+iBatch*self.nBatch].camera_src1.image
                batch.image_src2[iET, :, :, :] = vET[iET+iBatch*self.nBatch].camera_src2.image
                batch.prob_ref[iET, :, :, :] = vET[iET+iBatch*self.nBatch].camera_ref.heatmap
                batch.prob_src1[iET, :, :, :] = vET[iET+iBatch*self.nBatch].camera_src1.heatmap
                batch.prob_src2[iET, :, :, :] = vET[iET+iBatch*self.nBatch].camera_src2.heatmap
                batch.H1[iET, :, :] = vET[iET+iBatch*self.nBatch].H_stereo_rect1
                batch.H2[iET, :, :] = vET[iET+iBatch*self.nBatch].H_stereo_rect2
                batch.H01[iET, :, :] = vET[iET+iBatch*self.nBatch].H_stereo_rect01
                batch.H02[iET, :, :] = vET[iET+iBatch*self.nBatch].H_stereo_rect02
                batch.a1[iET] = vET[iET+iBatch*self.nBatch].a01
                batch.b1[iET] = vET[iET+iBatch*self.nBatch].b01
                batch.a2[iET] = vET[iET+iBatch*self.nBatch].a02
                batch.b2[iET] = vET[iET+iBatch*self.nBatch].b02
            self.vBatch.append(batch)

    def FetchData(self, i):
        r = i%len(self.vBatch)
        return self.vBatch[r]


class LabeledData(object):
    def __init__(self, nBatch, input_size, heatmap_size, nJoints):
        self.nBatch = nBatch
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.nJoints =  nJoints

    def SetData(self, vET):
        shuffle(vET)
        self.vBatch = []

        n = int(float(len(vET)) / self.nBatch)
        for iBatch in range(n):
            batch = Batch(self.nBatch, self.input_size, self.heatmap_size, self.nJoints)
            for iET in range(self.nBatch):
                batch.image_ref[iET, :, :, :] = vET[iET+iBatch*self.nBatch].camera_ref.image
                batch.prob_ref[iET, :, :, :] = vET[iET+iBatch*self.nBatch].camera_ref.heatmap
            self.vBatch.append(batch)

    def FetchData(self, i):
        r = i%len(self.vBatch)
        return self.vBatch[r]

class Triple(object):
    def __init__(self):
        self.H1 = np.zeros((3, 3))
        self.H01 = np.zeros((3, 3))
        self.H2 = np.zeros((3, 3))
        self.H02 = np.zeros((3, 3))
        self.a1 = 0
        self.b1 = 0
        self.a2 = 0
        self.b2 = 0

        self.ref = 0
        self.src1 = 0
        self.src2 = 0
        self.time = -1

class Pair(object):
    def __init__(self):
        self.H1 = np.zeros((3, 3))
        self.H01 = np.zeros((3, 3))
        self.a = 0
        self.b = 0

        self.ref = 0
        self.src = 0


class EpiTriple(object):
    def __init__(self):
        self.camera_ref = Camera()
        self.camera_src1 = Camera()
        self.camera_src2 = Camera()

        self.f0 = np.zeros(2)
        self.f1 = np.zeros(2)
        self.f2 = np.zeros(2)

    def SetParameter(self, camera_ref, camera_src1, camera_src2, heatmap_scale, ext):
        self.camera_ref = camera_ref
        self.camera_src1 = camera_src1
        self.camera_src2 = camera_src2

        self.heatmap_scale = heatmap_scale
        self.ext = ext

    def ComputeStereoRectification(self):
        # Homography for stereo rectification
        H1 = GetHomographyL(self.camera_src1.K, self.camera_src1.R, self.camera_src1.C, self.camera_ref.C, self.camera_ref.R[2, :])
        bb_x_1, bb_y_1 = self.GetShiftedOffset(H1, self.camera_src1.cropping_para)
        H1 = self.RectiNormalization(H1, self.camera_src1.cropping_para)
        H1_inv = LA.inv(H1)
        self.H_stereo_rect1 = H1_inv

        H2 = GetHomographyL(self.camera_src2.K, self.camera_src2.R, self.camera_src2.C, self.camera_ref.C, self.camera_ref.R[2, :])
        bb_x_2, bb_y_2 = self.GetShiftedOffset(H2, self.camera_src2.cropping_para)
        H2 = self.RectiNormalization(H2, self.camera_src2.cropping_para)
        H2_inv = LA.inv(H2)
        self.H_stereo_rect2 = H2_inv

        H01 = GetHomographyR(self.camera_ref.K, self.camera_ref.R, self.camera_ref.C, self.camera_src1.C, self.camera_ref.R[2, :])
        bb_x_01, bb_y_01 = self.GetShiftedOffset(H01, self.camera_ref.cropping_para)
        H01 = self.RectiNormalization(H01, self.camera_ref.cropping_para)
        self.H_stereo_rect01 = H01

        H02 = GetHomographyR(self.camera_ref.K, self.camera_ref.R, self.camera_ref.C, self.camera_src2.C, self.camera_ref.R[2, :])
        bb_x_02, bb_y_02 = self.GetShiftedOffset(H02, self.camera_ref.cropping_para)
        H02 = self.RectiNormalization(H02, self.camera_ref.cropping_para)
        self.H_stereo_rect02 = H02

        self.a01, self.b01 = self.GetMappingBtwRef_Src(self.camera_ref.K, self.camera_src1.K, bb_y_01, bb_y_1,
                                                       self.camera_ref.cropping_para, self.camera_src1.cropping_para)
        self.a02, self.b02 = self.GetMappingBtwRef_Src(self.camera_ref.K, self.camera_src2.K, bb_y_02, bb_y_2,
                                                       self.camera_ref.cropping_para, self.camera_src2.cropping_para)

    def GetMappingBtwRef_Src(self, K0, K1, bb0, bb1, cropping_para0, cropping_para1):
        a = cropping_para1[6] / cropping_para0[6] * K1[1, 1] / K0[1, 1]
        z = self.ext
        b = self.heatmap_scale * cropping_para1[6] * ((bb0[0] - K0[1, 2]) * K1[1, 1] / K0[1, 1] + K1[1, 2] - bb1[0]) + z - a*z
        return a, b

    def GetShiftedOffset(self, H, cropping_para):
        bb_x = cropping_para[0]
        bb_y = cropping_para[1]
        bb_center = np.zeros((3, 1))
        bb_center[0, 0] = (bb_x + cropping_para[2]) / 2
        bb_center[1, 0] = (bb_y + cropping_para[3]) / 2
        bb_center[2, 0] = 1
        bb_center_ = np.matmul(H, bb_center)
        bb_center_ = bb_center_ / bb_center_[2]

        w = cropping_para[2] - bb_x
        h = cropping_para[3] - bb_y

        bb_x_ = bb_center_[0] - w / 2
        bb_y_ = bb_center_[1] - h / 2
        return bb_x_, bb_y_

    def RectiNormalization(self, H, cropping_para):
        Ho_c = np.identity(3)  # from original image to cropped image
        cropped_scale = cropping_para[6]
        offset_x = cropping_para[4]
        offset_y = cropping_para[5]
        bb_x = cropping_para[0]
        bb_y = cropping_para[1]
        Ho_c[0, 0] = cropped_scale
        Ho_c[1, 1] = cropped_scale
        Ho_c[0, 2] = offset_x - cropped_scale * bb_x
        Ho_c[1, 2] = offset_y - cropped_scale * bb_y

        Hc_h = np.identity(3)  # from cropped image to heatmap
        Hc_h[0, 0] = self.heatmap_scale
        Hc_h[1, 1] = self.heatmap_scale

        bb_x_, bb_y_ = self.GetShiftedOffset(H, cropping_para)

        Hob_cb = np.identity(3)  # from original image to cropped image
        cropped_scale = cropping_para[6]
        offset_x = 0
        offset_y = 0
        bb_x = bb_x_
        bb_y = bb_y_
        Hob_cb[0, 0] = cropped_scale
        Hob_cb[1, 1] = cropped_scale
        Hob_cb[0, 2] = offset_x - cropped_scale * bb_x
        Hob_cb[1, 2] = offset_y - cropped_scale * bb_y

        Hcb_hb = np.identity(3)  # from cropped image to heatmap
        Hcb_hb[0, 0] = self.heatmap_scale
        Hcb_hb[1, 1] = self.heatmap_scale
        Hcb_hb[0, 2] = self.ext
        Hcb_hb[1, 2] = self.ext
        Hcb_hb[1, 2] = self.ext

        H1 = np.matmul(Hcb_hb, np.matmul(Hob_cb, np.matmul(H, np.matmul(LA.inv(Ho_c), LA.inv(Hc_h)))))
        return H1

class EpiSiamese(object):
    def __init__(self):
        self.camera_ref = Camera()
        self.camera_src = Camera()

        self.f0 = np.zeros(2)
        self.f1 = np.zeros(2)

    def SetParameter(self, camera_ref, camera_src1, heatmap_scale, ext):
        self.camera_ref = camera_ref
        self.camera_src = camera_src1

        self.heatmap_scale = heatmap_scale
        self.ext = ext

    def ComputeStereoRectification(self):
        # Homography for stereo rectification
        H1 = GetHomographyL(self.camera_src.K, self.camera_src.R, self.camera_src.C, self.camera_ref.C, self.camera_ref.R[2, :])
        bb_x_1, bb_y_1 = self.GetShiftedOffset(H1, self.camera_src.cropping_para)
        H1 = self.RectiNormalization(H1, self.camera_src.cropping_para)
        H1_inv = LA.inv(H1)
        self.H_stereo_rect1 = H1_inv

        H01 = GetHomographyR(self.camera_ref.K, self.camera_ref.R, self.camera_ref.C, self.camera_src.C, self.camera_ref.R[2, :])
        bb_x_01, bb_y_01 = self.GetShiftedOffset(H01, self.camera_ref.cropping_para)
        H01 = self.RectiNormalization(H01, self.camera_ref.cropping_para)
        H1_inv = LA.inv(H01)
        self.H_stereo_rect01 = H1_inv

        self.a01, self.b01 = self.GetMappingBtwRef_Src(self.camera_ref.K, self.camera_src.K, bb_y_01, bb_y_1,
                                                       self.camera_ref.cropping_para, self.camera_src.cropping_para)

    def GetMappingBtwRef_Src(self, K0, K1, bb0, bb1, cropping_para0, cropping_para1):
        a = cropping_para1[6] / cropping_para0[6] * K1[1, 1] / K0[1, 1]
        z = self.ext
        b = self.heatmap_scale * cropping_para1[6] * ((bb0[0] - K0[1, 2]) * K1[1, 1] / K0[1, 1] + K1[1, 2] - bb1[0]) + z - a*z
        return a, b

    def GetShiftedOffset(self, H, cropping_para):
        bb_x = cropping_para[0]
        bb_y = cropping_para[1]
        bb_center = np.zeros((3, 1))
        bb_center[0, 0] = (bb_x + cropping_para[2]) / 2
        bb_center[1, 0] = (bb_y + cropping_para[3]) / 2
        bb_center[2, 0] = 1
        bb_center_ = np.matmul(H, bb_center)
        bb_center_ = bb_center_ / bb_center_[2]

        w = cropping_para[2] - bb_x
        h = cropping_para[3] - bb_y

        bb_x_ = bb_center_[0] - w / 2
        bb_y_ = bb_center_[1] - h / 2
        return bb_x_, bb_y_

    def RectiNormalization(self, H, cropping_para):
        Ho_c = np.identity(3)  # from original image to cropped image
        cropped_scale = cropping_para[6]
        offset_x = cropping_para[4]
        offset_y = cropping_para[5]
        bb_x = cropping_para[0]
        bb_y = cropping_para[1]
        Ho_c[0, 0] = cropped_scale
        Ho_c[1, 1] = cropped_scale
        Ho_c[0, 2] = offset_x - cropped_scale * bb_x
        Ho_c[1, 2] = offset_y - cropped_scale * bb_y

        Hc_h = np.identity(3)  # from cropped image to heatmap
        Hc_h[0, 0] = self.heatmap_scale
        Hc_h[1, 1] = self.heatmap_scale

        bb_x_, bb_y_ = self.GetShiftedOffset(H, cropping_para)

        Hob_cb = np.identity(3)  # from original image to cropped image
        cropped_scale = cropping_para[6]
        offset_x = 0
        offset_y = 0
        bb_x = bb_x_
        bb_y = bb_y_
        Hob_cb[0, 0] = cropped_scale
        Hob_cb[1, 1] = cropped_scale
        Hob_cb[0, 2] = offset_x - cropped_scale * bb_x
        Hob_cb[1, 2] = offset_y - cropped_scale * bb_y

        Hcb_hb = np.identity(3)  # from cropped image to heatmap
        Hcb_hb[0, 0] = self.heatmap_scale
        Hcb_hb[1, 1] = self.heatmap_scale
        Hcb_hb[0, 2] = self.ext
        Hcb_hb[1, 2] = self.ext
        Hcb_hb[1, 2] = self.ext

        H1 = np.matmul(Hcb_hb, np.matmul(Hob_cb, np.matmul(H, np.matmul(LA.inv(Ho_c), LA.inv(Hc_h)))))
        return H1
