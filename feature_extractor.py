import numpy as np
import re
import os
import cv2
# import plot
# from tiles import tile_list
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# pd.options.display.max_rows = 100
from skimage.measure import label, regionprops_table
# from IPython.display import Image, clear_output
from skimage.morphology import dilation
from plotly.subplots import make_subplots
import plotly.io as pio
from skimage.transform import warp
pio.renderers.default = 'notebook'

class Image_fe:
    
    # reads image and applies median blur
    def __init__(self, img_name, path = r'C:\Users\adity\Documents\HWA\data\Crop', local = True):
        if local:
            self.path, self.img_name = path, img_name
            img = cv2.imread(f'{path}//{img_name}', 0)
        else:
            img, self.img_name = img_name, 'app image'
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        self.img = cv2.medianBlur(img, 5)
    
    # determins the positions in the image that corresponding to lines 
    def preprocess(self):
        lines, img = self.tile_list(), self.img
        self.lines = lines + ([img.shape[0]] if img.shape[0] != lines[-1] else [])
        self.label = self.label_cc()
    
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
        return minimum
    
    # word segmentation
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
        p25, p75 = (np.quantile(arr, i) for i in (0.25, 0.75))
        iqr = p75 - p25
        upper, lower = p75 + 1.5 * iqr, p25 - 1.5 * iqr
        return arr[(arr <= upper) & (arr >= lower)]

    
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
                cv2.arrowedLine(iline, (left, target), (right, target), color = 200, thickness = 2)
                cv2.arrowedLine(iline, (right, target), (left, target), color = 200, thickness = 2)
        self.space_img = img
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
    
    def slant_fe(self, li = False):
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

        temp = regionprops_table(self.label, properties = ['area'], extra_properties=[slant])
        slants, weights = temp['slant'], temp['area']
        self.slant_img = self.color(slants)
        slants = self.outlier(slants[weights > np.quantile(weights, 0.25)])
        series = pd.Series(
            data = [slants.mean(), slants.std(), self.entropy_bin(slants, 4)],
            index = ['slant_mean', 'slant_std', 'slant_entropy'],
            name = self.img_name
        )
        return temp['slant'] if li else series

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
        self.height_img = self.color(heights)
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
        

    # Area 
    def area_fe(self, ret = False):
        temp = regionprops_table(self.label, properties = ['area', 'bbox'])
        heights = temp['area'] / (temp['bbox-3'] - temp['bbox-1'])
        self.area_img = self.color(heights)
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
                self.space_fe(),
                self.space1_fe(),
                self.height_fe(),
                self.height1_fe(),
                self.height2_fe(),
                self.area_fe(),
                self.extent_fe(),
                self.solidity_fe()
            )
        )
    
    def color(self, li):
        img = -self.label.astype(float)
        for i, j in enumerate(li):
            img[img == -i-1] *= j/(-i-1)
        return img
    
    # def timepass(self):
    #     self.display(self.label, cmap = 'magma')
