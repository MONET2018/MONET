import tensorflow as tf
import numpy as np
import numpy.random
import cv2
from scipy.linalg import qr

from Epi_class import *
from numpy import linalg as LA

def VisualizeJointSortedHeatmap(image_gt, gt, image, heatmap, im_size, confidence):

    image_gt = cv2.resize(np.uint8(image_gt * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    heat = gt
    heat = heat / np.amax(heat, axis=(0, 1))
    heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    fin = np.clip(heat, 0, 1)
    fin = np.array(fin * 255, dtype=np.uint8)
    fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
    v_gt = cv2.addWeighted(image_gt, 0.5, fin, 0.5, 0)
    vis = v_gt

    for j in range(image.shape[0]):
        im = cv2.resize(np.uint8(image[j,:,:,:] * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        heat = heatmap[j,:,:]
        heat = heat / np.amax(heat, axis=(0, 1))
        heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        fin = np.clip(heat, 0, 1)
        fin = np.array(fin * 255, dtype=np.uint8)
        fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
        v = cv2.addWeighted(im, 0.5, fin, 0.5, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(v, "%0.2f"%(confidence[j]), (10, 30), font, 1, (255, 255, 255), 2, cv2.CV_AA)

        vis = np.concatenate((vis, v), axis=0)

    return vis

def VisualizeJointHeatmap1(image, heatmap, im_size, confidence):
    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    vis = []
    num_of_joints = heatmap.shape[2]
    for joint_id in range(len(confidence)):
        heat = heatmap[:, :, joint_id]
        heat = heat / np.amax(heat, axis=(0, 1))
        heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        fin = np.clip(heat, 0, 1)
        fin = np.array(fin * 255, dtype=np.uint8)
        fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
        v1 = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(v1, "%0.2f" % (confidence[joint_id]), (10, 30), font, 1, (255, 255, 255), 2, cv2.CV_AA)

        if (joint_id == 0):
            vis = v1
        else:
            vis = np.concatenate((vis, v1), axis=1)
    return vis

def VisualizeLimbSummarizedHeatmap(image, heatmap, im_size, heatmap_size):
    vis = []
    num_of_limbs = heatmap.shape[2] / 2
    agg = np.zeros((heatmap_size, heatmap_size,num_of_limbs))
    for joint_id in range(num_of_limbs):
        heat = heatmap[:, :, joint_id * 2] * heatmap[:, :, joint_id * 2] + heatmap[:, :, joint_id * 2 + 1] * heatmap[:,:,joint_id * 2 + 1]
        heat = np.sqrt(heat)
        agg[:,:,joint_id] = heat / np.amax(heat, axis=(0, 1))

    aggregated_map = np.amax(agg, axis=2)
    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)

    heat = aggregated_map
    heat = heat / np.amax(heat, axis=(0, 1))
    heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    fin = np.clip(heat, 0, 1)
    fin = np.array(fin * 255, dtype=np.uint8)
    fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
    v = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

    return v

def VisualizeJointSummarizedHeatmap(image, heatmap, im_size, heatmap_size, limb_link, markersize):
    aggregated_map = np.amax(heatmap[:,:,:-1], axis=2)

    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)

    heat = aggregated_map
    heat = heat / np.amax(heat, axis=(0, 1))
    heat = heat / np.amax(heat, axis=(0, 1))
    heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    fin = np.clip(heat, 0, 1)
    fin = np.array(fin * 255, dtype=np.uint8)
    fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
    v = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

    joint = []
    conf = np.zeros(heatmap.shape[2]-1)
    for j in range(heatmap.shape[2]-1):
        joint_id = j
        joint_heat = cv2.resize(heatmap[:, :, j], (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)

        joint_coord = np.unravel_index(np.argmax(joint_heat), (im_size, im_size))
        conf[j] = joint_heat[joint_coord[0], joint_coord[1]]
        # print(joint_coord)
        joint.append(joint_coord)
        cv2.circle(image, (joint_coord[1], joint_coord[0]), markersize, (255, 0, 0), -1)

    for j in range(len(limb_link)):
        cv2.line(image, (joint[limb_link[j][0]][1], joint[limb_link[j][0]][0]),
                 (joint[limb_link[j][1]][1], joint[limb_link[j][1]][0]), (255, 0, 0), markersize/2)

    vis = np.concatenate((v, image), axis=1)

    return vis, joint, conf

def VisualizeJointSingleHeatmap1(image, heatmap, im_size):
    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    vis = []

    heat = heatmap[:, :]
    heat = heat / np.amax(heat, axis=(0, 1))
    heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    fin = np.clip(heat, 0, 1)
    fin = np.array(fin * 255, dtype=np.uint8)
    fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
    v1 = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

    return v1

def VisualizeJointSingleHeatmap(image, heatmap, im_size, joint_id):
    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    vis = []

    heat = heatmap[:, :, joint_id]
    heat = heat / np.amax(heat, axis=(0, 1))
    heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    fin = np.clip(heat, 0, 1)
    fin = np.array(fin * 255, dtype=np.uint8)
    fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
    v1 = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

    return v1

def VisualizeJointHeatmap_confident_joint(image, heatmap, im_size, confident_joint):
    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    vis = []
    num_of_joints = heatmap.shape[2]
    for joint_id in range(num_of_joints):
        heat = heatmap[:, :, joint_id]
        heat = heat / np.amax(heat, axis=(0, 1))
        heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        fin = np.clip(heat, 0, 1)
        fin = np.array(fin * 255, dtype=np.uint8)
        fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
        v1 = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(v1, "%.02f" % (confident_joint[joint_id]), (10, 20), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)

        if (joint_id == 0):
            vis = v1
        else:
            vis = np.concatenate((vis, v1), axis=1)
    return vis

def VisualizeJointHeatmap_confident_joint_per_image(image, heatmap, im_size, confident_joint):

    vis = []
    num_of_joints = heatmap.shape[2]
    for joint_id in range(num_of_joints):
        im = cv2.resize(np.uint8(image[:,:,:,joint_id] * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        heat = heatmap[:, :, joint_id]
        heat = heat / np.amax(heat, axis=(0, 1))
        heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        fin = np.clip(heat, 0, 1)
        fin = np.array(fin * 255, dtype=np.uint8)
        fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
        v1 = cv2.addWeighted(im, 0.5, fin, 0.5, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(v1, "%.02f" % (confident_joint[joint_id]), (10, 20), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)

        if (joint_id == 0):
            vis = v1
        else:
            vis = np.concatenate((vis, v1), axis=1)
    return vis

def VisualizeJointHeatmap1(image, heatmap, im_size):
    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    vis = []
    num_of_joints = heatmap.shape[2]
    for joint_id in range(num_of_joints):
        heat = heatmap[:, :, joint_id]
        heat = heat / np.amax(heat, axis=(0, 1))
        heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        fin = np.clip(heat, 0, 1)
        fin = np.array(fin * 255, dtype=np.uint8)
        fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
        v1 = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

        if (joint_id == 0):
            vis = v1
        else:
            vis = np.concatenate((vis, v1), axis=0)
    return vis

def VisualizeJointHeatmap(image, heatmap, im_size):
    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    vis = []
    num_of_joints = heatmap.shape[2]
    for joint_id in range(num_of_joints):
        heat = heatmap[:, :, joint_id]
        heat = heat / np.amax(heat, axis=(0, 1))
        heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        fin = np.clip(heat, 0, 1)
        fin = np.array(fin * 255, dtype=np.uint8)
        fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
        v1 = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

        if (joint_id == 0):
            vis = v1
        else:
            vis = np.concatenate((vis, v1), axis=1)
    return vis



def VisualizeLimbHeatmap(image, heatmap, im_size):
    image = cv2.resize(np.uint8(image * 255), (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
    vis = []
    num_of_limbs = heatmap.shape[2]/2
    for joint_id in range(num_of_limbs):
        heat = heatmap[:, :, joint_id*2]*heatmap[:, :, joint_id*2]+heatmap[:, :, joint_id*2+1]*heatmap[:, :, joint_id*2+1]
        heat = np.sqrt(heat)
        heat = heat / np.amax(heat, axis=(0, 1))
        heat = cv2.resize(heat, (im_size, im_size), interpolation=cv2.INTER_LANCZOS4)
        fin = np.clip(heat, 0, 1)
        fin = np.array(fin * 255, dtype=np.uint8)
        fin = cv2.applyColorMap(fin, cv2.COLORMAP_JET)
        v1 = cv2.addWeighted(image, 0.5, fin, 0.5, 0)

        if (joint_id == 0):
            vis = v1
        else:
            vis = np.concatenate((vis, v1), axis=1)
    return vis

def BuildCameraProjectionMatrix(K, R, C):
    P = np.matmul(np.matmul(K, R), np.concatenate((np.eye(3), np.multiply(-1, C)), axis=1))
    return P

def Projection(P, X):
    x = np.matmul(P, np.concatenate((X, [1]), axis=0))
    x = np.multiply(x, 1. / x[2])
    return x

def GetTriple(vCamera):
    max_dist = 0
    for i in range(len(vCamera)-1):
        for j in range(i+1,len(vCamera)):
            dC = vCamera[i].C-vCamera[j].C
            if max_dist < LA.norm(dC):
                max_dist = LA.norm(dC)

    triple = []
    for iFrame in range(len(vCamera)):
        # print(iFrame)
        # print(len(triple))
        max_cos = 0
        idx1 = -1
        for iFrame1 in range(len(vCamera)):
            if iFrame == iFrame1:
                continue
            dC = vCamera[iFrame].C-vCamera[iFrame1].C
            dC /= LA.norm(dC)
            co = np.abs(np.dot(vCamera[iFrame].R[1,:], dC))
            if (max_cos < co) and (co > np.cos(10.0 * np.pi / 180.0)):
                max_cos = co
                idx1 = vCamera[iFrame1].frame

        min_dist = 100000.0
        idx2 = -1
        for iFrame1 in range(len(vCamera)):
            if iFrame == iFrame1:
                continue
            dC = vCamera[iFrame].C - vCamera[iFrame1].C
            dC1 = dC/LA.norm(dC)
            co = np.abs(np.dot(vCamera[iFrame].R[0, :], dC1))
            if (min_dist > LA.norm(dC)) and (co > np.cos(10.0 * np.pi / 180.0)):
                min_dist = LA.norm(dC)
                idx2 = vCamera[iFrame1].frame
        if idx2 != -1 and idx1 != -1:
            triple.append([vCamera[iFrame].frame, idx1, idx2])

        set = []
        for j in range(len(vCamera)):
            if iFrame == j:
                continue

            dC = vCamera[iFrame].C-vCamera[j].C
            dist = LA.norm(dC)
            if (dist < max_dist/2):
                set.append(j)


        triple1 = []
        for j1 in range(len(set)-1):
            for j2 in range(j1+1,len(set)):
                if (idx1 == vCamera[set[j1]].frame and idx2 == vCamera[set[j2]].frame) or (idx1 == vCamera[set[j2]].frame and idx2 == vCamera[set[j1]].frame):
                    continue
                v1 = vCamera[set[j1]].C - vCamera[iFrame].C
                v2 = vCamera[set[j2]].C - vCamera[iFrame].C
                v1 /= LA.norm(v1)
                v2 /= LA.norm(v2)
                c = v1[0,0]*v2[0,0]+v1[1,0]*v2[1,0]+v1[2,0]*v2[2,0]
                if (c < np.cos(np.pi/3)) and (c > np.cos(np.pi/3*2.0)):
                    triple1.append([vCamera[iFrame].frame, vCamera[set[j1]].frame, vCamera[set[j2]].frame])

        if len(triple1) > 10:
            np.random.shuffle(triple1)
            for i in range(10):
                triple.append(triple1[i])
        else:
            for i in range(len(triple1)):
                triple.append(triple1[i])

    return triple

def GetTriple1(vCamera):
    max_dist = 0
    for i in range(len(vCamera)-1):
        for j in range(i+1,len(vCamera)):
            dC = vCamera[i].C-vCamera[j].C
            if max_dist < LA.norm(dC):
                max_dist = LA.norm(dC)

    triple = []
    for iFrame in range(len(vCamera)):
        # print(iFrame)
        # print(len(triple))
        max_cos = 0
        idx1 = -1
        triple2 = []
        for iFrame1 in range(len(vCamera)):
            if iFrame == iFrame1:
                continue
            dC = vCamera[iFrame].C-vCamera[iFrame1].C
            dC /= LA.norm(dC)
            co = np.abs(np.dot(vCamera[iFrame].R[1,:], dC))
            if (max_cos < co) and (co > np.cos(10.0 * np.pi / 180.0)):
                max_cos = co
                idx1 = vCamera[iFrame1].frame

        min_dist = 100000.0
        idx2 = -1
        for iFrame1 in range(len(vCamera)):
            if iFrame == iFrame1:
                continue
            dC = vCamera[iFrame].C - vCamera[iFrame1].C
            dC1 = dC/LA.norm(dC)
            co = np.abs(np.dot(vCamera[iFrame].R[0, :], dC1))
            if (min_dist > LA.norm(dC)) and (co > np.cos(10.0 * np.pi / 180.0)):
                min_dist = LA.norm(dC)
                idx2 = vCamera[iFrame1].frame
        if idx2 != -1 and idx1 != -1:
            triple2.append([vCamera[iFrame].frame, idx1, idx2])

        set = []
        for j in range(len(vCamera)):
            if iFrame == j:
                continue

            dC = vCamera[iFrame].C-vCamera[j].C
            dist = LA.norm(dC)
            if (dist < max_dist/3):
                set.append(j)


        triple1 = []
        for j1 in range(len(set)-1):
            for j2 in range(j1+1,len(set)):
                if (idx1 == vCamera[set[j1]].frame and idx2 == vCamera[set[j2]].frame) or (idx1 == vCamera[set[j2]].frame and idx2 == vCamera[set[j1]].frame):
                    continue
                v1 = vCamera[set[j1]].C - vCamera[iFrame].C
                v2 = vCamera[set[j2]].C - vCamera[iFrame].C

                v1_x = np.dot(vCamera[iFrame].R[0, :], v1)
                v1_y = np.dot(vCamera[iFrame].R[1, :], v1)
                v2_x = np.dot(vCamera[iFrame].R[0, :], v2)
                v2_y = np.dot(vCamera[iFrame].R[1, :], v2)

                v1_norm = np.sqrt(v1_x * v1_x + v1_y * v1_y)
                v2_norm = np.sqrt(v2_x * v2_x + v2_y * v2_y)
                v1_x /= v1_norm
                v1_y /= v1_norm
                v2_x /= v2_norm
                v2_y /= v2_norm

                c = v1_x*v2_x + v1_y*v2_y
                if (c < np.cos(np.pi/4)) and (c > np.cos(np.pi/4*3.0)):
                    triple1.append([vCamera[iFrame].frame, vCamera[set[j1]].frame, vCamera[set[j2]].frame])

        triple_new = triple2
        np.random.shuffle(triple1)
        idx = 0
        while len(triple_new) < 10 and idx < len(triple1):
            isGood = True
            for i in range(len(triple_new)):
                if triple1[idx][1] == triple_new[i][1] or triple1[idx][2] == triple_new[i][1] \
                      or triple1[idx][1] == triple_new[i][2] or triple1[idx][2] == triple_new[i][2]:
                    isGood = False
            if isGood:
                triple_new.append(triple1[idx])

            idx += 1

        #
        # if len(triple1) > 10:
        #     np.random.shuffle(triple1)
        #     for i in range(10):
        #         triple_new.append(triple1[i])
        # else:
        #     for i in range(len(triple1)):
        #         triple_new.append(triple1[i])
        if (len(triple_new)==0):
            continue
        triple.append(triple_new)

    return triple


def GetPair(vCamera):
    max_dist = 0
    for i in range(len(vCamera)-1):
        for j in range(i+1,len(vCamera)):
            dC = vCamera[i].C-vCamera[j].C
            if max_dist < LA.norm(dC):
                max_dist = LA.norm(dC)

    print('max dist %f' % max_dist)
    pair = []
    for iFrame in range(len(vCamera)):
        pair2 = np.zeros((len(vCamera), 4))
        for iFrame1 in range(len(vCamera)):
            if (iFrame1 == iFrame):
                continue
            dC = vCamera[iFrame].C - vCamera[iFrame1].C
            dist = LA.norm(dC)
            dC /= LA.norm(dC)

            v1_x = np.dot(vCamera[iFrame].R[0, :], dC)
            v1_y = np.dot(vCamera[iFrame].R[1, :], dC)
            pair2[iFrame1,0] = dist
            pair2[iFrame1,1] = np.arctan2(v1_y, v1_x)
            pair2[iFrame1,2] = vCamera[iFrame].frame
            pair2[iFrame1,3] = vCamera[iFrame1].frame

        dist_ord = np.argsort(pair2[:,0])

        pair1 = []
        pair1.append(pair2[dist_ord[1],:])
        pair1.append(pair2[dist_ord[2],:])
        pair1.append(pair2[dist_ord[3],:])

        set = []
        for i in range(len(vCamera)):
            if i == dist_ord[0] or i == dist_ord[1] or i == dist_ord[2] or i == dist_ord[3]:
                continue
            if pair2[i,0] > max_dist/3:
                continue
            # if pair2[i,0] < max_dist/6:
            #     continue
            set.append(pair2[i,:])

        np.random.shuffle(set)

        set1 = pair1
        for i in range(len(set)):
            isGood = True
            for j in range(len(set1)):
                if abs(set1[j][1]-set[i][1]) < np.pi/10:
                    isGood = False
            if (isGood==True):
                set1.append(set[i])
        pair.append(set1)

    return pair

def GetNearViewCamera(vCamera, ref, threshold):
    set = []
    for iCamera in range(len(vCamera)):
        c = np.dot(vCamera[iCamera].R[2,:], vCamera[ref].R[2,:])
        if (c > np.cos(threshold)):
            set.append(iCamera)

    return set

def GetConfidentView(heatmap, threshold):
    heatmap_size = heatmap.shape[1]
    conf = np.zeros((heatmap.shape[0]))
    joint = np.zeros((heatmap.shape[0], 2))
    for iView in range(heatmap.shape[0]):
        norml2 = LA.norm(heatmap[iView,:,:])
        heatmap[iView,:,:] = heatmap[iView,:,:] / norml2
        joint_coord = np.unravel_index(np.argmax(heatmap[iView,:,:]), (heatmap_size, heatmap_size))
        conf[iView] = heatmap[iView, joint_coord[0], joint_coord[1]]
        joint[iView,:] = [joint_coord[1], joint_coord[0]]
    # confident_view = np.zeros((heatmap.shape[0]))

    idx = np.argsort(conf)
    n = len(idx)
    confident_view = conf
    # n = 20
    # print(conf)
    # for iView in range(n):
    #     # if conf[idx[n-iView-1]] > threshold:
    #     confident_view[idx[n-iView-1]] = conf[idx[n-iView-1]]#1
    return confident_view, idx, joint

def Image2Heatmap(x, scale_im2crop, scale_crop2heat, bb, off):
    X = np.zeros(2)
    X[0] = (x[0]/scale_crop2heat - off[0])/scale_im2crop + bb[0]
    X[1] = (x[1] / scale_crop2heat - off[1]) / scale_im2crop + bb[1]
    return X

def Heatmap2Image(X, scale_im2crop, scale_crop2heat, bb, off):
    x = np.zeros(2)
    x[0] = scale_crop2heat * (scale_im2crop * (X[0]-bb[0]) + off[0])
    x[1] = scale_crop2heat * (scale_im2crop * (X[1] - bb[1]) + off[1])
    return x

def Vec2Skew(x):
    skew_p = np.array(([0, -1, x[1]], [1, 0, -x[0]], [-x[1], x[0], 0]))
    return skew_p

def SolveMinNonzero(A, b):
    x1, res, rnk, s = LA.lstsq(A, b)
    if rnk == A.shape[1]:
        return x1   # nothing more to do if A is full-rank
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    Z = Q[:, rnk:].conj()
    C = np.linalg.solve(Z[rnk:], -x1[rnk:])
    return x1 + Z.dot(C)

def TriangulatePoint(P1, P2, x1, x2):
    A = np.zeros((6, 3))
    b = np.zeros((6, 1))

    skew_p = Vec2Skew(np.array([x1[0], x1[1],1]))
    A[:3, :] = np.matmul(skew_p, P1[:, 0:3])
    b[:3, :] = np.reshape(np.dot(-skew_p, P1[:, 3]), (3, 1))

    skew_p = Vec2Skew(np.array([x2[0], x2[1],1]))

    A[3:6, :] = np.matmul(skew_p, P2[:, 0:3])
    b[3:6, :] = np.reshape(np.dot(-skew_p, P2[:, 3]), (3, 1))

    Point3d, _, _, _ = LA.lstsq(A, b)

    #print(A)
    #print(b)

    #Point3d = SolveMinNonzero(A, b)

    return Point3d

def TriangulatePointMany(vP, x):
    A = np.zeros((3*len(vP), 3))
    b = np.zeros((3*len(vP), 1))

    for i in range(len(vP)):

        skew_p = Vec2Skew(np.array([x[i,0], x[i,1],1]))
        A[3*i:3*(i+1), :] = np.matmul(skew_p, vP[i][:, 0:3])
        b[3*i:3*(i+1), :] = np.reshape(np.dot(-skew_p, vP[i][:, 3]), (3, 1))
    #
    # print(A)
    # print(b)

    TriangulatePointPoint3d, _, _, _ = LA.lstsq(A, b)

    # print(Point3d)

    Point3d = SolveMinNonzero(A, b)

    return Point3d

def RANSAC_Triangulation(vP, vx, nIters):
    nMaxInliers = 0
    X = np.zeros(3)
    threshold = 20
    for i in range(nIters):
        rand_idx = np.random.permutation(len(vP))
        Xr = TriangulatePoint(vP[rand_idx[0]], vP[rand_idx[1]], vx[rand_idx[0],:], vx[rand_idx[1],:])
        Xr = np.array([Xr[0,0], Xr[1,0], Xr[2,0]])

        # Xr = [Xr, 1]
        nInlier = 0

        for iP in range(len(vP)):
            x = Projection(vP[iP], Xr)
            #
            # rp = np.dot(vP[iP], Xr)
            # rp = [rp[0] / rp[2], rp[1] / rp[2]]
            err = LA.norm(x[:2] - vx[iP,:])
            if err < threshold:
                nInlier += 1

        if (nInlier > nMaxInliers):

            nMaxInliers = nInlier
            X = Xr[:3]

        if (nMaxInliers > np.float(len(vP)*0.8)):
            break

    # print(nMaxInliers)
    return X, nMaxInliers

    # M_ind = [project_index.index(i) for i in idx]
    # M_matrices = [project_matrix[i] for i in M_ind]
    # A = np.zeros((3 * k, 3))
    # b = np.zeros((3 * k, 1))
    # for i in range(0, k):
    #     p = [sorted_loc[i][0], sorted_loc[i][1], 1.0]
    #     skew_p = np.array(([0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]))
    #     A[3 * i:3 * (i + 1), :] = np.dot(skew_p, M_matrices[i][:, 0:3])
    #     b[3 * i:3 * (i + 1), :] = np.reshape(np.dot(-skew_p, M_matrices[i][:, 3]), (3, 1))
    # Point3d = SolveMinNonzero(A, b)
    # Point3d = [Point3d[0][0], Point3d[1][0], Point3d[2][0], 1.0]
    # proj_3d.append(Point3d)
    # error = 0.0
    # for z in range(0, k):
    #     p = sorted_loc[z]
    #     rp = np.dot(M_matrices[z], Point3d)
    #     rp = [rp[0] / rp[2], rp[1] / rp[2]]
    #     proj_2d.append((int(rp[0]), int(rp[1])))
    #     err = np.linalg.norm(np.array(rp) - np.array(p))
    #     error = error + err
    # joint_error = joint_error + error

def RerankConfidentView(confidence, idx_camera):
    for i in range(len(confidence)):
        isIn = False
        for j in range(len(idx_camera)):
            if i == idx_camera[j]:
                isIn = True

        if isIn == False:
            confidence[i] = 0.0

    idx = np.argsort(confidence)
    # print(confidence)
    # print(idx)

    return confidence, idx

def GetTriple_exhaustive(vCamera):
    max_dist = 0
    for i in range(len(vCamera)-1):
        for j in range(i+1,len(vCamera)):
            dC = vCamera[i].C-vCamera[j].C
            if max_dist < LA.norm(dC):
                max_dist = LA.norm(dC)
    triple = []
    for iFrame in range(len(vCamera)):
        set = []
        for iFrame1 in range(len(vCamera)):
            if iFrame == iFrame1:
                continue
            dC = vCamera[iFrame].C-vCamera[iFrame1].C
            if (max_dist/2 > LA.norm(dC)):
                set.append(iFrame1)

        triple1 = []
        for j1 in range(len(set)-1):
            for j2 in range(j1+1,len(set)):
                v1 = vCamera[set[j1]].C - vCamera[iFrame].C
                v2 = vCamera[set[j2]].C - vCamera[iFrame].C
                v1 /= LA.norm(v1)
                v2 /= LA.norm(v2)
                # print(v1)
                # print(v2)
                c = v1[0,0]*v2[0,0]+v1[1,0]*v2[1,0]+v1[2,0]*v2[2,0]
                # print(c)
                if (c < np.cos(np.pi/3)) and (c > np.cos(np.pi/3*2.0)):
                    triple1.append([vCamera[iFrame].frame, vCamera[set[j1]].frame, vCamera[set[j2]].frame])
        triple.append(triple1)
        # print(len(set))
        # print(len(triple1))

    return triple

def GetHomographyL(K1, R1, C1, C2, rz):
    rx = C2 - C1
    rx = np.multiply(rx, 1. / LA.norm(rx))
    # sign = np.matmul(R1[0, :], rx)
    # if sign < 0:
    #     rx = np.multiply(rx, -1)
    rz = rz.reshape((3, 1))
    rz = rz - np.multiply(np.matmul(rz.T, rx), rx)
    rz = np.multiply(rz, 1. / LA.norm(rz))
    ry = np.matmul(Vec2Skew(rz), rx)
    Rt = np.concatenate((rx.T, ry.T, rz.T), axis=0)
    Kinv = LA.inv(K1)
    H = np.matmul(K1, np.matmul(Rt, np.matmul(R1.T, Kinv)))
    return H


def GetHomographyR(K1, R1, C1, C2, rz):
    rx = C1 - C2
    rx = np.multiply(rx, 1. / LA.norm(rx))
    # sign = np.matmul(R1[0, :], rx)
    # if sign < 0:
    #     rx = np.multiply(rx, -1)
    rz = rz.reshape((3, 1))
    rz = rz - np.multiply(np.matmul(rz.T, rx), rx)
    rz = np.multiply(rz, 1. / LA.norm(rz))
    ry = np.matmul(Vec2Skew(rz), rx)
    Rt = np.concatenate((rx.T, ry.T, rz.T), axis=0)
    Kinv = LA.inv(K1)
    H = np.matmul(K1, np.matmul(Rt, np.matmul(R1.T, Kinv)))
    return H

def GetHomography_vert(K1, R1, C1, C2, rz):
    ry = C2 - C1
    ry = np.multiply(ry, 1. / LA.norm(ry))
    sign = np.matmul(R1[1, :], ry)
    if sign < 0:
        ry = np.multiply(ry, -1)
    rz = rz.reshape((3, 1))
    rz = rz - np.multiply(np.matmul(rz.T, ry), ry)
    rz = np.multiply(rz, 1. / LA.norm(rz))
    rx = np.matmul(Vec2Skew(ry), rz)
    Rt = np.concatenate((rx.T, ry.T, rz.T), axis=0)
    Kinv = LA.inv(K1)
    H = np.matmul(K1, np.matmul(Rt, np.matmul(R1.T, Kinv)))
    return H

def Vec2Skew(x):
    skew = np.zeros((3, 3))
    skew[0, 1] = -x[2]
    skew[1, 0] = x[2]
    skew[0, 2] = x[1]
    skew[2, 0] = -x[1]
    skew[1, 2] = -x[0]
    skew[2, 1] = x[0]
    return skew

def GenerateGaussianIntensity(x0, sigma, im_size):
    u_x, u_y = np.meshgrid(range(0, im_size[1]), range(0, im_size[0]))
    intensity = np.exp(-((u_x - x0[0]) ** 2 + (u_y - x0[1]) ** 2) / sigma / sigma)
    return intensity

def Get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def Get_pixel_value_1D(img, x):
    shape = tf.shape(x)
    batch_size = shape[0]
    length = shape[1]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    b = tf.tile(batch_idx, (1, length))

    indices = tf.stack([b, x], 2)

    return tf.gather_nd(img, indices)


def Linear_sampler(array, x):
    B = tf.shape(array)[0]
    L = tf.shape(array)[1]

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, L - 1)
    x1 = tf.clip_by_value(x1, 0, L - 1)

    # get pixel value at corner coords
    Ia = Get_pixel_value_1D(array, x0)
    Ib = Get_pixel_value_1D(array, x1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')

    # calculate deltas
    wa = 1 - (x - x0)
    wb = 1 - (x1 - x)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=2)
    wb = tf.expand_dims(wb, axis=2)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib])

    return out


def Bilinear_sampler(img, x, y):
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, W - 1)
    x1 = tf.clip_by_value(x1, 0, W - 1)
    y0 = tf.clip_by_value(y0, 0, H - 1)
    y1 = tf.clip_by_value(y1, 0, H - 1)

    # get pixel value at corner coords
    Ia = Get_pixel_value(img, x0, y0)
    Ib = Get_pixel_value(img, x0, y1)
    Ic = Get_pixel_value(img, x1, y0)
    Id = Get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


def GridGenerator(width, height, homography):
    num_batch = tf.shape(homography)[0]
    x_t, y_t = tf.meshgrid(tf.range(0, width), tf.range(0, height))

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    homography = tf.cast(homography, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(homography, sampling_grid)
    out = tf.zeros((num_batch, 2, width * height))
    out1 = tf.reshape(tf.div(batch_grids[:, 0, :], batch_grids[:, 2, :]), [num_batch, 1, width * height])
    out2 = tf.reshape(tf.div(batch_grids[:, 1, :], batch_grids[:, 2, :]), [num_batch, 1, width * height])
    out = tf.concat((out1, out2), axis=1)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    out = tf.reshape(out, [num_batch, 2, height, width])

    return out

def IntensityWarping(intensity, homography_inv):
    B = tf.shape(intensity)[0]
    H = tf.shape(intensity)[1]
    W = tf.shape(intensity)[2]
    C = tf.shape(intensity)[3]

    grid = GridGenerator(W, H, homography_inv)

    x_s = grid[:, 0, :, :]
    y_s = grid[:, 1, :, :]

    warped = Bilinear_sampler(intensity, x_s, y_s)

    return warped

def IntensityWarping_ext(intensity, homography_inv, ext):
    B = tf.shape(intensity)[0]
    H = tf.shape(intensity)[1]+ext*2
    W = tf.shape(intensity)[2]+ext*2
    C = tf.shape(intensity)[3]

    grid = GridGenerator(W, H, homography_inv)

    x_s = grid[:, 0, :, :]
    y_s = grid[:, 1, :, :]

    warped = Bilinear_sampler(intensity, x_s, y_s)

    return warped


def EpipolarTransfer(prob, H1, H01, f0, f1):
    B = tf.shape(prob)[0]
    H = tf.shape(prob)[1]
    W = tf.shape(prob)[2]
    C = tf.shape(prob)[3]

    warped = IntensityWarping(prob, H1)
    rowmax = tf.reduce_max(warped, axis=2)
    u = tf.range(H)
    u = tf.cast(u, 'float32')
    u = tf.expand_dims(u, axis=0)
    u = tf.tile(u, tf.stack([B, 1]))

    f0_0 = tf.tile(tf.expand_dims(f0[:, 0], axis=1), tf.stack([1, H]))
    f0_1 = tf.tile(tf.expand_dims(f0[:, 1], axis=1), tf.stack([1, H]))

    f1_0 = tf.tile(tf.expand_dims(f1[:, 0], axis=1), tf.stack([1, H]))
    f1_1 = tf.tile(tf.expand_dims(f1[:, 1], axis=1), tf.stack([1, H]))

    u1 = (u - f0_1) * f1_0 / f0_0 + f1_1
    rowmax = Linear_sampler(rowmax, u1)
    rowmax = tf.expand_dims(rowmax, axis=2)
    rowmax = tf.tile(rowmax, tf.stack([1, 1, W, 1]))
    out = IntensityWarping(rowmax, H01)

    return out

def EpipolarTransferHeatmap(prob, H1, H01, a, b):
    B = tf.shape(prob)[0]
    H = tf.shape(prob)[1]
    W = tf.shape(prob)[2]
    C = tf.shape(prob)[3]

    warped = IntensityWarping(prob, H1)
    rowmax = tf.reduce_max(warped, axis=2)
    u = tf.range(H)
    u = tf.cast(u, 'float32')
    u = tf.expand_dims(u, axis=0)
    u = tf.tile(u, tf.stack([B, 1]))

    a = tf.tile(tf.expand_dims(a[:,0], axis=1), tf.stack([1, H]))
    b = tf.tile(tf.expand_dims(b[:,0], axis=1), tf.stack([1, H]))

    # u1 = (u-b)/a
    u1 = a*u+b
    # u1 = u

    rowmax2 = rowmax
    rowmax = Linear_sampler(rowmax, u1)
    rowmax = tf.expand_dims(rowmax, axis=2)
    rowmax = tf.tile(rowmax, tf.stack([1, 1, W, 1]))
    out = IntensityWarping(rowmax, H01)

    return out

def EpipolarTransferHeatmap_siamese_src(prob, H1, a, b, ext_size):
    B = tf.shape(prob)[0]
    H = tf.shape(prob)[1]+2*ext_size
    W = tf.shape(prob)[2]+2*ext_size
    C = tf.shape(prob)[3]

    warped = IntensityWarping_ext(prob, H1, ext_size)
    rowmax = tf.reduce_max(warped, axis=2)
    u = tf.range(H)
    u = tf.cast(u, 'float32')
    u = tf.expand_dims(u, axis=0)
    u = tf.tile(u, tf.stack([B, 1]))

    a = tf.tile(tf.expand_dims(a[:,0], axis=1), tf.stack([1, H]))
    b = tf.tile(tf.expand_dims(b[:,0], axis=1), tf.stack([1, H]))

    # u1 = (u-b)/a
    u1 = a*u+b
    # u1 = u

    rowmax2 = rowmax
    rowmax = Linear_sampler(rowmax, u1)
    # rowmax = tf.expand_dims(rowmax, axis=2)
    # rowmax = tf.tile(rowmax, tf.stack([1, 1, W, 1]))
    # out = IntensityWarping_ext(rowmax, H01, -ext_size)

    return rowmax

def EpipolarTransferHeatmap_siamese_ref(prob, H1, ext_size):
    B = tf.shape(prob)[0]
    H = tf.shape(prob)[1]+2*ext_size
    W = tf.shape(prob)[2]+2*ext_size
    C = tf.shape(prob)[3]

    warped = IntensityWarping_ext(prob, H1, ext_size)
    rowmax = tf.reduce_max(warped, axis=2)
    u = tf.range(H)
    u = tf.cast(u, 'float32')
    u = tf.expand_dims(u, axis=0)
    u = tf.tile(u, tf.stack([B, 1]))

    # a = tf.tile(tf.expand_dims(a[:,0], axis=1), tf.stack([1, H]))
    # b = tf.tile(tf.expand_dims(b[:,0], axis=1), tf.stack([1, H]))

    # u1 = (u-b)/a
    # u1 = a*u+b
    # u1 = u

    rowmax2 = rowmax
    rowmax = Linear_sampler(rowmax, u)
    # rowmax = tf.expand_dims(rowmax, axis=2)
    # rowmax = tf.tile(rowmax, tf.stack([1, 1, W, 1]))
    # out = IntensityWarping_ext(rowmax, H01, -ext_size)

    return rowmax


def EpipolarTransferHeatmap_ext(prob, H1, H01, a, b, ext_size):
    B = tf.shape(prob)[0]
    H = tf.shape(prob)[1]+2*ext_size
    W = tf.shape(prob)[2]+2*ext_size
    C = tf.shape(prob)[3]

    warped = IntensityWarping_ext(prob, H1, ext_size)
    rowmax = tf.reduce_max(warped, axis=2)
    u = tf.range(H)
    u = tf.cast(u, 'float32')
    u = tf.expand_dims(u, axis=0)
    u = tf.tile(u, tf.stack([B, 1]))

    a = tf.tile(tf.expand_dims(a[:,0], axis=1), tf.stack([1, H]))
    b = tf.tile(tf.expand_dims(b[:,0], axis=1), tf.stack([1, H]))

    # u1 = (u-b)/a
    u1 = a*u+b
    # u1 = u

    rowmax2 = rowmax
    rowmax = Linear_sampler(rowmax, u1)
    rowmax = tf.expand_dims(rowmax, axis=2)
    rowmax = tf.tile(rowmax, tf.stack([1, 1, W, 1]))
    out = IntensityWarping_ext(rowmax, H01, -ext_size)

    return out


def EpipolarTransferHeatmap_ext_softmax(prob, H1, H01, a, b, ext_size):
    B = tf.shape(prob)[0]
    H = tf.shape(prob)[1]+2*ext_size
    W = tf.shape(prob)[2]+2*ext_size
    C = tf.shape(prob)[3]

    warped = IntensityWarping_ext(prob, H1, ext_size)
    rowmax = 100*tf.reduce_max(warped, axis=2)
    rowmax = tf.nn.softmax(rowmax, dim=1)
    u = tf.range(H)
    u = tf.cast(u, 'float32')
    u = tf.expand_dims(u, axis=0)
    u = tf.tile(u, tf.stack([B, 1]))

    a = tf.tile(tf.expand_dims(a[:,0], axis=1), tf.stack([1, H]))
    b = tf.tile(tf.expand_dims(b[:,0], axis=1), tf.stack([1, H]))

    # u1 = (u-b)/a
    u1 = a*u+b
    # u1 = u

    rowmax2 = rowmax
    rowmax = Linear_sampler(rowmax, u1)

    rowmax = tf.expand_dims(rowmax, axis=2)
    rowmax = tf.tile(rowmax, tf.stack([1, 1, W, 1]))
    out = IntensityWarping_ext(rowmax, H01, -ext_size)

    return out
