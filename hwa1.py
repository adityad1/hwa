import numpy as np
# import re
# import os
import cv2
import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
from skimage.measure import label, regionprops_table
from skimage.morphology import dilation
# from plotly.subplots import make_subplots
# import plotly.io as pio
from skimage.transform import warp
from sklearn.mixture import GaussianMixture
import warnings
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import AffineTransform, warp
from skimage.measure import shannon_entropy
# pio.renderers.default = 'notebook'

class Image_fe:
    
    # reads image and applies median blur
    def __init__(self, img_name, path = r'C:/Users/jahna/Downloads/HQA/Crop/Crop', local = True, ruled = False):
        if local:
            self.path, self.img_name = path, img_name
            img = cv2.imread(f'{path}//{img_name}', 0)
        else:
            img, self.img_name = img_name, 'app image'
        if ruled:
            self.img = self.remove_rules(img)
            if not ruled:
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                self.img = cv2.medianBlur(img, 5)
        else:
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            self.img = cv2.medianBlur(img, 5)
        self.ruled = ruled
    
    # determins the positions in the image that corresponding to lines 
    def preprocess(self):
        lines, img = self.tile_list(), self.img
        self.lines = lines
        self.label = self.label_words()
        if (np.unique(self.label).shape[0] <= len(lines) * 2) or self.ruled:
            self.label = self.label_cc()
    
    # ruled lines removal
    def remove_rules(self, img):
        gray, src_img = img.copy(), img.copy()
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, binarized_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        convert_r = lambda angle: np.pi * angle / 180
        convert_d = lambda angle: 180 * angle / np.pi
        width = binarized.shape[1] // 6
        select_block = lambda image, r, c: image[width*(r-1): width*r, width*(c-1): width*c]
        boxes_hough = []
        for i, j in [(3, 2), (3, 5), (5, 2), (5, 5)]:
            _, theta, distance = hough_line_peaks(
                *hough_line(select_block(binarized_inv, i, j)),
                threshold = (width * 2) // 3
            )
            boxes_hough.append(list(map(
                lambda y: (round(convert_d(y[0])) + ((y[0]<0)*180), y[1] * (y[0]//abs(y[0]))),
                zip(theta, distance)
            )))
        line_angles = [j[0] for i in map(lambda box: box[:2], boxes_hough) for j in i]
        if not all(boxes_hough):
            self.ruled = False
            return img
        votes = {angle: sum(map(lambda y: angle-3 <= y <= angle+3, line_angles)) for angle in set(line_angles)}
        for angle in line_angles:
            votes[angle] += 1
        line_angle = max(votes, key = lambda y: votes[y])
        matrix = np.eye(3)
        matrix[1][0] = convert_r(line_angle-90)
        hline_img = warp(
            src_img,
            matrix,
            mode = 'wrap',
            preserve_range = True
        )
        hline_img = np.uint8(hline_img)

        box_distances = []
        for box in boxes_hough:
            hlines = sorted([distance for angle, distance in box if line_angle-3 <= angle <= line_angle+3])
            distances = [hlines[idx+1] - hlines[idx] for idx in range(len(hlines)-1)]
            box_distances.append(np.mean(distances))
        dist = np.mean(box_distances, dtype = 'int')

#         gray = cv2.cvtColor(hline_img, cv2.COLOR_RGB2GRAY)
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, binarized_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        _, theta, distance = hough_line_peaks(*hough_line(select_block(binarized_inv, 1, 2)), threshold = (width * 2) // 3)
        upper_box = list(map(lambda y: (round(convert_d(y[0])) + ((y[0]<0)*180), y[1] * (y[0]//abs(y[0]))), (zip(theta, distance))))
        first_line = int(sorted(filter(lambda y: 87 <= y[0] <= 93, upper_box), key = lambda y: y[1])[0][1])

        test_subject, width = binarized_inv.copy(), gray.shape[1] // 6
        vline_strips = range(first_line - dist // 2, test_subject.shape[0] - dist, dist)
        for vstrip_idx in vline_strips:
            for hstrip_idx in range(0, 6*width, width):
                test_strip = test_subject[vstrip_idx: vstrip_idx+dist, hstrip_idx: hstrip_idx + width]
                hough, strip_theta, strip_dist = hough_line(test_strip)
                thresh, line_theta, line_dist = hough_line_peaks(hough, strip_theta, strip_dist, threshold = test_strip.shape[1] * (3/4), num_peaks = 1)
                if line_dist.shape[0] == 0:
                    continue
                line_dist, line_theta, r, theta = (hough.shape[0] // 2) + int(line_dist[0]), round(convert_d(line_theta[0])) + 90, line_dist, line_theta
                cos, sin, w = np.cos(theta), np.sin(theta), test_strip.shape[1]
                thickness = (hough[line_dist-5:line_dist+5, line_theta] > (test_strip.shape[1] / 2)).sum()
                cv2.line(test_strip, (0, int(r/sin)), (w, int((r-(w*cos)) / sin)), 0, thickness)
                if shannon_entropy(test_strip) < 0.15:
                    test_strip *= 0
                test_strip = cv2.dilate(cv2.medianBlur(test_strip, ksize = 3), kernel = np.ones((10, 1)))
                test_subject[vstrip_idx: vstrip_idx+dist, hstrip_idx: hstrip_idx + width] = test_strip
        return test_subject
        
    # performs line segementation 
    def tile_list(self):
        hp = (self.img == 255).sum(axis = 1)
        arr = pd.Series(hp).rolling(30, min_periods = 1).sum().to_numpy()
        minima, switch = [[]], arr < arr.mean()
        for i in range(1, len(arr)):
            if switch[i] != switch[i-1]:
                if switch[i]:
                    minima.append([i])
                else:
                    minima[-1].append(i)
        minimum = [np.argmin(hp[:minima[0][0]])]
        for i in minima[1:-1]:
            minimum.append(np.argmin(hp[i[0]:i[1]]) + i[0])
        minimum.append(minima[-1][0] + np.argmin(hp[minima[-1][0]:]))
        return ([0] if minimum[0] != 0 else []) + minimum + ([self.img.shape[0]] if minimum[-1] != self.img.shape[0] else [])
    
    # word segmentation v2
    def label_words(self):
        lines, gaps, words_segmented, used, flag, img = self.lines, [], np.empty(shape = (0, self.img.shape[1]), dtype = 'int'), 0, False, self.img.copy()
        def label_new(img):
            if img.sum():
                dilated, line = img.copy(), img
                if line.sum():
                    line_vp, gaps, count = np.trim_zeros(line.sum(axis= 0)), [], 0
                    for i in line_vp:
                        if not i:
                            count += 1
                        elif count:
                            gaps.append(count)
                            count = 0
                    gaps = np.array(gaps)
                    kernel = np.ones(shape = (50, 2))
                    dilated = dilation(dilated, kernel)

                limg = (img == 255).astype('uint8')
                reg = pd.DataFrame(
                    regionprops_table(label(dilated), properties = ('label', 'bbox', 'image', 'area'))
                ).set_index('label')
                c = 1
                for i, minr, minc, maxr, maxc, mask, area in reg.itertuples():
                    limg[minr: maxr, minc: maxc][mask] *= c
                    c += 1
                return limg
            else:
                return img

        warnings.filterwarnings('ignore')
        for line_no in range(len(lines)-1):
            line = label_new(img[lines[line_no]: lines[line_no+1]]).astype('int')
            line[line > 0] += used
            words_segmented, used = np.vstack((words_segmented, line)), line.max()
            max_index = img[lines[line_no]: lines[line_no+1]].sum(axis = 1).argmax()
            df = pd.DataFrame(regionprops_table(line, properties = ['label', 'centroid', 'bbox']))
            df.set_index('label', inplace = True)
            df.sort_values(by = 'centroid-1', inplace = True)
            for i in range(df.shape[0]-1):
                x, y = df.iloc[i].name, df.iloc[i+1].name
                t, b = max(df.loc[x, 'bbox-0'], df.loc[y, 'bbox-0']), min(df.loc[x, 'bbox-2'], df.loc[y, 'bbox-2'])
                left_x, left_y, right_x, right_y = df.loc[x, 'bbox-1'], df.loc[y, 'bbox-1'], df.loc[x, 'bbox-3'], df.loc[y, 'bbox-3']
                x_line, y_line = line == x, line == y
                words_mask = (x_line[t:b].sum(axis = 1) & y_line[t:b].sum(axis = 1)) != 0
                if (left_x < left_y < right_y < right_x) or (left_y < left_x < right_x < right_y):
                    gaps.append((-1, x, y, 0))
                elif (t >= b) or ((words_mask != 0).sum() == 0):
                    gaps.append((left_y - right_x, x, y, 0))
                else:
                    target = np.argmin([np.argmax(y_line[i]) + np.argmax(x_line[i][::-1]) for i in np.arange(t, b)[words_mask]])
                    left, right = 2400-np.argmax(x_line[t:b][words_mask][target][::-1]), np.argmax(y_line[t:b][words_mask][target])
                    gaps.append((right - left, x, y, 0))
            else:
                gaps.append((0, 0, 0, 0))
        gaps, n = np.array(gaps, dtype = 'int'), 1
        gap_mask = (gaps[:, 0] > 0)
        gm = GaussianMixture(n_components = 2, random_state = 0)
        gm = gm.fit(np.vstack((gaps[:, :n][gap_mask], np.zeros((10, n)))))
        if (~gap_mask).sum():
            return self.label_cc()
        
        gap_predictions = gm.predict(gaps[:, :n][gap_mask])
        gap_label = np.argmin([gaps[gap_mask][gap_predictions == i][:, 0].mean() for i in range(2)])
        gaps[:, -1][gap_mask] = gap_predictions
        gaps[:, -1][~gap_mask] = gap_label


        prev = gaps[0, 2]
        for d, x, y, p in gaps:
            if d:
                prev = prev if prev else x
                if p == gap_label:
                    words_segmented[words_segmented == y] = prev
                else:
                    prev = y
            else:
                prev = 0
        iterations = 0
        while (gap_label in gap_predictions) and gap_mask.sum():
            img, gaps = words_segmented.copy(), []
            for line_no in range(len(lines)-1):
                line = img[lines[line_no]: lines[line_no+1]]
                df = pd.DataFrame(regionprops_table(line, properties = ['label', 'centroid', 'bbox']))
                df.set_index('label', inplace = True)
                df.sort_values(by = 'bbox-1', inplace = True)
                for i in range(df.shape[0]-1):
                    x, y = df.iloc[i].name, df.iloc[i+1].name
                    t, b = max(df.loc[x, 'bbox-0'], df.loc[y, 'bbox-0']), min(df.loc[x, 'bbox-2'], df.loc[y, 'bbox-2'])
                    left_x, left_y, right_x, right_y = df.loc[x, 'bbox-1'], df.loc[y, 'bbox-1'], df.loc[x, 'bbox-3'], df.loc[y, 'bbox-3']
                    x_line, y_line = line == x, line == y
                    words_mask = (x_line[t:b].sum(axis = 1) & y_line[t:b].sum(axis = 1)) != 0
                    if (left_x < left_y < right_y < right_x) or (left_y < left_x < right_x < right_y):
                        gaps.append((-1, x, y, 0))
                    elif (t >= b) or ((words_mask != 0).sum() == 0):
                        gaps.append((left_y - right_x, x, y, 0))
                    else:
                        target = np.argmin([np.argmax(y_line[i]) + np.argmax(x_line[i][::-1]) for i in np.arange(t, b)[words_mask]])
                        left, right = 2400-np.argmax(x_line[t:b][words_mask][target][::-1]), np.argmax(y_line[t:b][words_mask][target])
                        gaps.append((right - left, x, y, 0))
                else:
                    gaps.append((0, 0, 0, 0))
            gaps = np.array(gaps, dtype = 'int')
            gap_mask = (gaps[:, 0] > 0)
            if gap_mask.sum():
                gap_predictions = gm.predict(gaps[:, :n][gap_mask])
                gaps[:, -1][gap_mask] = gap_predictions
            elif (~gap_mask).sum():
                gaps[:, -1][~gap_mask] = gap_label
            prev = gaps[0, 2]
            for d, x, y, p in gaps:
                if d:
                    prev = prev if prev else x
                    if p == gap_label:
                        words_segmented[words_segmented == y] = prev
                    else:
                        prev = y
                else:
                    prev = 0
            iterations += 1
            if iterations == 10:
                break
        return words_segmented

    # word segmentation v1
    def label_cc(self):
        lines, dilated, img = self.lines, self.img.copy(), self.img
        for line_no in range(len(lines)-1):
            line = img[lines[line_no]: lines[line_no+1]]
            if line.sum():
                line_vp, gaps, count = np.trim_zeros(line.sum(axis= 0)), [], 0
                for i in line_vp:
                    if not i:
                        count += 1
                    elif count:
                        gaps.append(count)
                        count = 0
                gaps = np.array(gaps)
                kernel_size = int(np.ceil(np.append([10], gaps[(gaps < 60) + (gaps > 5)]).mean()))
                kernel = np.ones(shape = (1, kernel_size))
                dilated[lines[line_no]: lines[line_no+1]] = dilation(dilated[lines[line_no]: lines[line_no+1]], kernel)
        
        limg = (img == 255).astype('uint8')
        reg = pd.DataFrame(
            regionprops_table(label(dilated), properties = ('label', 'bbox', 'image', 'area'))
        ).set_index('label')
        c = 1
        for i, minr, minc, maxr, maxc, mask, area in reg.itertuples():
            limg[minr: maxr, minc: maxc][mask] *= c
            c += 1
        return limg
    
    def outlier(self, arr):
#         print(arr.shape)
        p15 = np.quantile(arr, 0.15)
        return arr[arr >= p15]
    
    def entropy_bin(self, data, bin_width, round_off = False):
        if data.shape[0] == 0:
            return np.nan
        if round_off:
            cut = pd.cut(
                data,
                bins = np.arange(data.min() - bin_width, data.max(), bin_width)
            )
        else:
            cut = pd.cut(
                data,
                bins = np.arange(round(data.min()) - bin_width, round(data.max()), bin_width)
            )
        n = cut.shape[0]
        counts = cut.value_counts()
        return counts.apply(lambda v: -v * np.log2(v/n) / n).sum()

    # determines space between words and stores it in an array
    def space_fe(self):
        img, lines, label, space_list, space_img = self.img, self.lines, self.label, [], self.img.copy()
        for line_no in range(len(lines)-1):
            line, iline = label[lines[line_no]: lines[line_no+1]], img[lines[line_no]: lines[line_no+1]]
            max_index = iline.sum(axis = 1).argmax()
            df = pd.DataFrame(regionprops_table(line, properties = ['label', 'centroid', 'bbox']))
            df.set_index('label', inplace = True)
            df.sort_values(by = 'centroid-1', inplace = True)
            df = df[df['bbox-0'] < max_index]
            df = df[max_index < df['bbox-2']]
            for i in range(df.shape[0]-1):
                x, y = df.iloc[i].name, df.iloc[i+1].name
                t, b = max(df.loc[x, 'bbox-0'], df.loc[y, 'bbox-0']), min(df.loc[x, 'bbox-2'], df.loc[y, 'bbox-2'])
                x_line, y_line = line == x, line == y
                target = np.argmin([np.argmax(y_line[i]) + np.argmax(x_line[i][::-1]) for i in range(t, b)]) + t
                left, right = line.shape[1]-np.argmax(x_line[target][::-1]), np.argmax(y_line[target])
                space_list.append(right - left)
        #         cv2.arrowedLine(iline, (left, target), (right, target), color = 200, thickness = 2)
        #         cv2.arrowedLine(iline, (right, target), (left, target), color = 200, thickness = 2)
        # self.space_img = img
        space_list = self.outlier(np.array(space_list))
        series = pd.Series(
            data = [space_list.mean(), space_list.std(), self.entropy_bin(space_list, 10)],
            index = ['space_mean', 'space_std', 'space_entropy'],
            name = self.img_name
        )
        return series
    
    # space centroid 
    def space1_fe(self):
        img, lines, label, space_list = self.img, self.lines, self.label, []
        for line_no in range(len(lines)-1):
            line, iline = label[lines[line_no]: lines[line_no+1]], img[lines[line_no]: lines[line_no+1]]
            max_index = iline.sum(axis = 1).argmax()
            df = pd.DataFrame(regionprops_table(line, properties = ['label', 'centroid', 'bbox']))
            df.set_index('label', inplace = True)
            df.sort_values(by = 'centroid-1', inplace = True)
            for i in range(df.shape[0]-1):
                x, y = df.iloc[i]['centroid-1'], df.iloc[i+1]['centroid-1']
                x_width , y_width = (df.iloc[i]['bbox-3'] - df.iloc[i]['bbox-1']) ,  (df.iloc[i+1]['bbox-3'] - df.iloc[i+1]['bbox-1'])
                centroid_dist = y - x - (x_width + y_width) / 2 
                space_list.append(centroid_dist)
        space_list = self.outlier(np.array(space_list))
        space_list = space_list[space_list >= 0]
        series = pd.Series(
            data = [space_list.std() , space_list.mean(), self.entropy_bin(space_list, 10)],
            index = ['space1_std', 'space1_mean', 'space1_entropy'],
            name = self.img_name
        )
        return series   
    
    # determines space between words and stores it in an array (modified)
    def space2_fe(self):
        img, lines, label, space_list, space_img = self.img, self.lines, self.label, [], self.img.copy()
        for line_no in range(len(lines)-1):
            line, iline = label[lines[line_no]: lines[line_no+1]], img[lines[line_no]: lines[line_no+1]]
            max_index = iline.sum(axis = 1).argmax()
            df = pd.DataFrame(regionprops_table(line, properties = ['label', 'centroid', 'bbox']))
            df.set_index('label', inplace = True)
            df.sort_values(by = 'centroid-1', inplace = True)
            df = df[df['bbox-0'] < max_index]
            df = df[max_index < df['bbox-2']]
            for i in range(df.shape[0]-1):
                x, y = df.iloc[i].name, df.iloc[i+1].name
                t, b = max(df.loc[x, 'bbox-0'], df.loc[y, 'bbox-0']), min(df.loc[x, 'bbox-2'], df.loc[y, 'bbox-2'])
                x_line, y_line = line == x, line == y
                words_mask = (x_line[t:b].sum(axis = 1) & y_line[t:b].sum(axis = 1)) != 0
                # left, right = line.shape[1]-np.argmax(x_line[target][::-1]), np.argmax(y_line[target])
                if (t >= b) or ((words_mask != 0).sum() == 0):
                    continue
                target = np.argmin([np.argmax(y_line[i]) + np.argmax(x_line[i][::-1]) for i in np.arange(t, b)[words_mask]])
                left, right = 2400-np.argmax(x_line[t:b][words_mask][target][::-1]), np.argmax(y_line[t:b][words_mask][target])
                space_list.append(right - left)
                cv2.arrowedLine(iline, (left, target), (right, target), color = 200, thickness = 2)
                cv2.arrowedLine(iline, (right, target), (left, target), color = 200, thickness = 2)
        self.space_img = img
        space_list = self.outlier(np.array(space_list))
        series = pd.Series(
            data = [space_list.mean(), space_list.std(), self.entropy_bin(space_list, 10)],
            index = ['space2_mean', 'space2_std', 'space2_entropy'],
            name = self.img_name
        )
        return series
    
    def slant_fe(self, ret = False):
        # extracts the slant for each word 
        def slant(word):
            hp, vp, scores = word.sum(axis = 1), word.sum(axis = 0), []
            f, b = np.argmax(hp>0), word.shape[0] - np.argmax(hp[::-1]>0)
            l, r = np.argmax(vp>0), word.shape[1] - np.argmax(vp[::-1]>0)
            word = word[f:b+1, l:r+1]
            matrix, scores, w = np.eye(3), [], word.shape[0]
            for i in np.linspace(-1, 1, 91):
                matrix[0][1] = i
                word_sheared = warp(word, matrix, mode='wrap')
                trim, vp = [w - np.argmax(col) - np.argmax(col[::-1]) for col in word_sheared.T], word_sheared.sum(axis = 0)
                scores.append((vp[trim == vp]**2).sum())
            return 45 + np.argmax(scores)

        temp = regionprops_table(self.label, properties = ['label', 'area'], extra_properties=[slant])
        slants, weights = temp['slant'], temp['area']
        self.slant_img = self.color(slants, temp['label'])
        slants = self.outlier(slants[weights > np.quantile(weights, 0.25)])
        series = pd.Series(
            data = [slants.mean(), slants.std(), self.entropy_bin(slants, 4)],
            index = ['slant_mean', 'slant_std', 'slant_entropy'],
            name = self.img_name
        )
        return temp['slant'] if ret else series

    # height feature extractor
    def height_fe(self, ret = False):
        def height(word):
            li = np.apply_along_axis(lambda col: word.shape[0] - np.argmax(col) if col.sum() else 0, 0, word)
            return li.mean() + li.std()
        temp = regionprops_table(self.label, properties = ['area'], extra_properties = [height])
        heights, weights = temp['height'], temp['area']
        heights = self.outlier(heights[weights > np.quantile(weights, 0.25)])
        series = pd.Series(
            data = [heights.mean(), heights.std(), self.entropy_bin(heights, 10)],
            index = ['height_mean', 'height_std', 'height_entropy'],
            name = self.img_name
        )
        return temp['height'] if ret else series
    
    # height1 feature extractor
    def height1_fe(self, ret = False):
        temp = regionprops_table(self.label, properties = ['bbox', 'area'])
        heights, weights = temp['bbox-2'] - temp['bbox-0'], temp['area']
        # self.height_img = self.color(heights)
        heights = self.outlier(heights[weights > np.quantile(weights, 0.25)])
        series = pd.Series(
            data = [heights.mean(), heights.std(), self.entropy_bin(heights, 10)],
            index = ['height1_mean', 'height1_std', 'height1_entropy'],
            name = self.img_name
        )
        return temp['bbox-2'] - temp['bbox-0'] if ret else series
    
    def height2_fe(self, ret = False):
        lines = pd.Series(self.lines)
        lines = (lines - lines.shift()).iloc[:-1].dropna()
        series = pd.Series(
            data = [lines.mean(), lines.std(), self.entropy_bin(lines, 10)],
            index = ['height2_mean', 'height2_std', 'height2_entropy'],
            name = self.img_name
        )
        return lines if ret else series
    
    # improved height function
    def height3_fe(self, ret = False):
        def height(img):
            hp = np.sum(img, axis=1)
            max_pos = np.argmax(hp)
            threshold = hp[max_pos]/3
            b, e = max_pos, max_pos
            while (hp[b] > threshold and b != 0) or (hp[e]> threshold and e != len(hp)-1):
                if hp[b] > threshold and (b > 0):
                    b -= 1
                if hp[e] > threshold and (e < len(hp)-1):
                    e+=1
            return e-b
        temp = regionprops_table(self.label, properties = ['label', 'area'], extra_properties = [height])
        heights, weights = temp['height'], temp['area']
        self.height_img = self.color(heights, temp['label'])
        heights = heights[weights > np.quantile(weights, 0.1)]
        series = pd.Series(
            data = [heights.mean(), heights.std(), self.entropy_bin(heights, 10)],
            index = ['height3_mean', 'height3_std', 'height3_entropy'],
            name = self.img_name
        )
        return heights if ret else series

        

    # Area 
    def area_fe(self, ret = False):
        temp = regionprops_table(self.label, properties = ['area', 'bbox', 'label'])
        heights = temp['area'] / (temp['bbox-3'] - temp['bbox-1'])
        self.area_img = self.color(heights, temp['label'])
        weights, h = temp['area'], heights
        heights = self.outlier(heights[weights > np.quantile(weights, 0.25)])
        series = pd.Series(
            data = [heights.mean(), heights.std(), self.entropy_bin(heights, 2, round_off = True)],
            index = ['area_mean', 'area_std', 'area_entropy'],
            name = self.img_name
        )
        return h if ret else series
    
    # Solidity 
    def solidity_fe(self, ret = False):
        temp = regionprops_table(self.label, properties = ['area', 'solidity'])
        weights, heights = temp['area'], temp['solidity']
        heights = self.outlier(heights[weights > np.quantile(weights, 0.25)])
        series = pd.Series(
            data = [heights.mean(), heights.std(), self.entropy_bin(heights, 0.1, round_off = True)],
            index = ['solidity_mean', 'solidity_std', 'solidity_entropy'],
            name = self.img_name
        )
        return heights if ret else series
    
    # Extent 
    def extent_fe(self, ret = False):
        temp = regionprops_table(self.label, properties = ['area', 'extent'])
        weights, heights = temp['area'], temp['extent']
        heights = self.outlier(heights[weights > np.quantile(weights, 0.25)])
        series = pd.Series(
            data = [heights.mean(), heights.std(), self.entropy_bin(heights, 0.1, round_off = True)],
            index = ['extent_mean', 'extent_std', 'extent_entropy'],
            name = self.img_name
        )
        return heights if ret else series

    def word_fe(self):
        self.preprocess()
        return pd.concat(
            (
                self.slant_fe(),
                # self.space_fe(),
                self.space1_fe(),
                self.space2_fe(),
                self.height_fe(),
                self.height1_fe(),
                # self.height2_fe(),
                self.height3_fe(),
                self.area_fe(),
                self.extent_fe(),
                self.solidity_fe()
            )
        )
    
    def color(self, li, labels):
        img = self.label.astype(float).copy()
        for i, j in zip(labels, li):
            img[img == i] *= j / i
        return img
    
