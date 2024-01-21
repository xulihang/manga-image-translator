import cv2
from functools import reduce
from DBNet_resnet34 import TextDetection
from model_ocr import OCR
from model_ocr_ctc import OCR as OCRCTC
from model_ocr_48px import OCR as OCR48PX
from inpainting_aot import AOTGenerator
import einops
import imgproc
import numpy as np
import craft_utils
import dbnet_utils
import torch
from text_mask_utils import filter_masks, complete_mask
from utils import AvgMeter, BBox, Quadrilateral, image_resize, quadrilateral_can_merge_region
from typing import List, Tuple
from ocr_utils import count_valuable_text, generate_text_direction, merge_bboxes_text_region,resize_keep_aspect, overlay_mask

class Translator():

    def __init__(self, use_cuda=False,img_detect_size=1536,unclip_ratio=2.2,box_threshold=0.7,text_threshold=0.5, inpainting_size=2048):
        self.dictionary = None
        self.model_ocr = None
        self.model_detect = None
        self.model_inpainting = None
        self.use_48px_model = False
        self.use_ctc_model = False
        self.use_cuda = use_cuda
        self.img_detect_size = img_detect_size
        self.unclip_ratio = unclip_ratio
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.inpainting_size = inpainting_size
        
        
    def load_ocr_model(self):
        if self.dictionary == None:
            if self.use_48px_model == True:
                with open('alphabet-all-v7.txt', 'r', encoding='utf-8') as fp :
                    self.dictionary = [s[:-1] for s in fp.readlines()]
            else:
                with open('alphabet-all-v5.txt', 'r', encoding='utf-8') as fp :
                    self.dictionary = [s[:-1] for s in fp.readlines()]
            
            if self.use_48px_model == True:
                model = OCR48PX(self.dictionary, 768)
                sd = torch.load('ocr_ar_48px.ckpt')
                model.load_state_dict(sd)
                model.eval()
                if self.use_cuda :
                    model = model.cuda()
                self.model_ocr = model
            elif self.use_ctc_model == True:
                model = OCRCTC(self.dictionary, 768)
                sd = torch.load('ocr-ctc.ckpt', map_location = 'cpu')
                sd = sd['model'] if 'model' in sd else sd
                del sd['encoders.layers.0.pe.pe']
                del sd['encoders.layers.1.pe.pe']
                del sd['encoders.layers.2.pe.pe']
                model.load_state_dict(sd, strict = False)
                model.eval()
                if self.use_cuda :
                    model = model.cuda()
                self.model_ocr = model
            else:
                model = OCR(self.dictionary, 768)
                model.load_state_dict(torch.load('ocr.ckpt', map_location='cpu'))
                model.eval()
                if self.use_cuda :
                    model = model.cuda()
                self.model_ocr = model


    def load_detect_model(self):
        if self.model_detect == None:
            model = TextDetection()
            sd = torch.load('detect.ckpt', map_location='cpu')
            model.load_state_dict(sd['model'] if 'model' in sd else sd)
            model.eval()
            if self.use_cuda :
                model = model.cuda()
            self.model_detect = model

    def load_inpainting_model(self):
        if self.model_inpainting == None:
            model = AOTGenerator()
            sd = torch.load('inpainting.ckpt', map_location='cpu')
            model.load_state_dict(sd['gen'] if 'gen' in sd else sd)
            model.eval()
            if self.use_cuda :
                model = model.cuda()
            self.model_inpainting = model
        
    def detect(self,img):
        self.load_detect_model()
        img_resized, target_ratio, _, pad_w, pad_h = imgproc.resize_aspect_ratio(img, self.img_detect_size, cv2.INTER_LINEAR, mag_ratio = 1)
        ratio_h = ratio_w = 1 / target_ratio
        img_np_resized = img_resized.astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img_np_resized)
        if self.use_cuda :
            img = img.cuda()
        img = einops.rearrange(img, 'h w c -> 1 c h w')
        with torch.no_grad():
            db, mask = self.model_detect(img)
            db = db.sigmoid().cpu()
            mask = mask[0, 0, :, :].cpu().numpy()
            mask = (mask * 255.0).astype(np.uint8)
            det = dbnet_utils.SegDetectorRepresenter(self.text_threshold, self.box_threshold, unclip_ratio = self.unclip_ratio)
            boxes, scores = det({'shape':[(img_resized.shape[0], img_resized.shape[1])]}, db)
            boxes, scores = boxes[0], scores[0]
            if boxes.size == 0 :
                polys = []
            else :
                idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
                polys, _ = boxes[idx], scores[idx]
                polys = polys.astype(np.float64)
                polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 1)
                polys = polys.astype(np.int16)
            textlines = [Quadrilateral(pts.astype(int), '', 0) for pts in polys]
        return textlines, mask

    def run_ocr(self, img, quadrilaterals: List[Tuple[Quadrilateral, str]], dictionary, model, max_chunk_size = 2) :
        text_height = 32
        regions = [q.get_transformed_region(img, d, text_height) for q, d in quadrilaterals]
        out_regions = []
        perm = sorted(range(len(regions)), key = lambda x: regions[x].shape[1])
        ix = 0
        for indices in self.chunks(perm, max_chunk_size) :
            N = len(indices)
            widths = [regions[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
            for i, idx in enumerate(indices) :
                W = regions[idx].shape[1]
                region[i, :, : W, :] = regions[idx]
                #cv2.imwrite(f'ocrs/{ix}.png', region[i, :, :, :])
                ix += 1
            images = (torch.from_numpy(region).float() - 127.5) / 127.5
            images = einops.rearrange(images, 'N H W C -> N C H W')
            ret = self.ocr_infer_bacth(images, model, widths)
            for i, (pred_chars_index, prob, fr, fg, fb, br, bg, bb) in enumerate(ret) :
                if prob < 0.6 :
                    continue
                fr = (torch.clip(fr.view(-1), 0, 1).mean() * 255).long().item()
                fg = (torch.clip(fg.view(-1), 0, 1).mean() * 255).long().item()
                fb = (torch.clip(fb.view(-1), 0, 1).mean() * 255).long().item()
                br = (torch.clip(br.view(-1), 0, 1).mean() * 255).long().item()
                bg = (torch.clip(bg.view(-1), 0, 1).mean() * 255).long().item()
                bb = (torch.clip(bb.view(-1), 0, 1).mean() * 255).long().item()
                seq = []
                for chid in pred_chars_index :
                    ch = dictionary[chid]
                    if ch == '<S>' :
                        continue
                    if ch == '</S>' :
                        break
                    if ch == '<SP>' :
                        ch = ' '
                    seq.append(ch)
                txt = ''.join(seq)
                print(prob, txt, f'fg: ({fr}, {fg}, {fb})', f'bg: ({br}, {bg}, {bb})')
                cur_region = quadrilaterals[indices[i]][0]
                cur_region.text = txt
                cur_region.prob = prob
                cur_region.fg_r = fr
                cur_region.fg_g = fg
                cur_region.fg_b = fb
                cur_region.bg_r = br
                cur_region.bg_g = bg
                cur_region.bg_b = bb
                out_regions.append(cur_region)
        return out_regions

    def run_ocr_ctc(self, img, quadrilaterals: List[Tuple[Quadrilateral, str]], dictionary, model, max_chunk_size = 2) :
        text_height = 48
        regions = [q.get_transformed_region(img, d, text_height) for q, d in quadrilaterals]
        out_regions = []
        perm = sorted(range(len(regions)), key = lambda x: regions[x].shape[1])
        ix = 0
        for indices in self.chunks(perm, max_chunk_size) :
            N = len(indices)
            widths = [regions[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
            for i, idx in enumerate(indices) :
                W = regions[idx].shape[1]
                region[i, :, : W, :] = regions[idx]
                #cv2.imwrite(f'ocrs/{ix}.png', region[i, :, :, :])
                ix += 1
            images = (torch.from_numpy(region).float() - 127.5) / 127.5
            images = einops.rearrange(images, 'N H W C -> N C H W')
            ret = self.ocr_infer_bacth(images, model, widths)
            for i, single_line in enumerate(ret):
                total_fr = AvgMeter()
                total_fg = AvgMeter()
                total_fb = AvgMeter()
                total_br = AvgMeter()
                total_bg = AvgMeter()
                total_bb = AvgMeter()
                total_logprob = AvgMeter()
                seq = []
                for (childIndex, logprob, fr, fg, fb, br, bg, bb) in single_line:
                    ch = dictionary[childIndex]
                    if ch == '<S>' :
                        continue
                    if ch == '</S>' :
                        break
                    if ch == '<SP>' :
                        ch = ' '
                    total_logprob(logprob)
                    if ch != ' ':
                        total_fr(int(fr * 255))
                        total_fg(int(fg * 255))
                        total_fb(int(fb * 255))
                        total_br(int(br * 255))
                        total_bg(int(bg * 255))
                        total_bb(int(bb * 255))
                    seq.append(ch)
                txt = ''.join(seq)
                prob = np.exp(total_logprob())
                if prob < 0.5:
                    continue
                fr = int(total_fr())
                fg = int(total_fg())
                fb = int(total_fb())
                br = int(total_br())
                bg = int(total_bg())
                bb = int(total_bb())
                print(prob, txt, f'fg: ({fr}, {fg}, {fb})', f'bg: ({br}, {bg}, {bb})')
                cur_region = quadrilaterals[indices[i]][0]
                cur_region.text = txt
                cur_region.prob = prob
                cur_region.fg_r = fr
                cur_region.fg_g = fg
                cur_region.fg_b = fb
                cur_region.bg_r = br
                cur_region.bg_g = bg
                cur_region.bg_b = bb
                out_regions.append(cur_region)
        return out_regions

    def run_ocr_48px(self, img, quadrilaterals: List[Tuple[Quadrilateral, str]], dictionary, model, max_chunk_size = 2) :
        text_height = 48
        regions = [q.get_transformed_region(img, d, text_height) for q, d in quadrilaterals]
        out_regions = []
        perm = sorted(range(len(regions)), key = lambda x: regions[x].shape[1])
        ix = 0
        for indices in self.chunks(perm, max_chunk_size) :
            N = len(indices)
            widths = [regions[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
            for i, idx in enumerate(indices) :
                W = regions[idx].shape[1]
                region[i, :, : W, :] = regions[idx]
                #cv2.imwrite(f'ocrs/{ix}.png', region[i, :, :, :])
                ix += 1
            images = (torch.from_numpy(region).float() - 127.5) / 127.5
            images = einops.rearrange(images, 'N H W C -> N C H W')
            ret = self.ocr_infer_bacth(images, model, widths)
            for i, (pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred) in enumerate(ret):
                if prob < 0.2:
                    continue
                has_fg = (fg_ind_pred[:, 1] > fg_ind_pred[:, 0])
                has_bg = (bg_ind_pred[:, 1] > bg_ind_pred[:, 0])
                seq = []
                fr = AvgMeter()
                fg = AvgMeter()
                fb = AvgMeter()
                br = AvgMeter()
                bg = AvgMeter()
                bb = AvgMeter()
                for chid, c_fg, c_bg, h_fg, h_bg in zip(pred_chars_index, fg_pred, bg_pred, has_fg, has_bg) :
                    ch = self.model_ocr.dictionary[chid]
                    if ch == '<S>':
                        continue
                    if ch == '</S>':
                        break
                    if ch == '<SP>':
                        ch = ' '
                    seq.append(ch)
                    if h_fg.item() :
                        fr(int(c_fg[0] * 255))
                        fg(int(c_fg[1] * 255))
                        fb(int(c_fg[2] * 255))
                    if h_bg.item() :
                        br(int(c_bg[0] * 255))
                        bg(int(c_bg[1] * 255))
                        bb(int(c_bg[2] * 255))
                    else :
                        br(int(c_fg[0] * 255))
                        bg(int(c_fg[1] * 255))
                        bb(int(c_fg[2] * 255))
                txt = ''.join(seq)
                fr = min(max(int(fr()), 0), 255)
                fg = min(max(int(fg()), 0), 255)
                fb = min(max(int(fb()), 0), 255)
                br = min(max(int(br()), 0), 255)
                bg = min(max(int(bg()), 0), 255)
                bb = min(max(int(bb()), 0), 255)
                print(f'prob: {prob} {txt} fg: ({fr}, {fg}, {fb}) bg: ({br}, {bg}, {bb})')
                cur_region = quadrilaterals[indices[i]][0]
                if isinstance(cur_region, Quadrilateral):
                    cur_region.text = txt
                    cur_region.prob = prob
                    cur_region.fg_r = fr
                    cur_region.fg_g = fg
                    cur_region.fg_b = fb
                    cur_region.bg_r = br
                    cur_region.bg_g = bg
                    cur_region.bg_b = bb
                else:
                    cur_region.text.append(txt)
                    cur_region.update_font_colors(np.array([fr, fg, fb]), np.array([br, bg, bb]))

                out_regions.append(cur_region)
        return out_regions
        
    def ocr_infer_bacth(self, img, model, widths) :
        if self.use_cuda :
            img = img.cuda()
        with torch.no_grad():
            if self.use_48px_model == True:
                return model.infer_beam_batch(img, widths, beams_k = 5, max_seq_length = 255)
            elif self.use_ctc_model == True:
                return model.decode(img, widths, 0, verbose = True)
            else:
                return model.infer_beam_batch(img, widths, beams_k = 5, max_seq_length = 255)
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def ocr(self,img, img_bbox,textlines):
        self.load_ocr_model()
        if self.use_48px_model == True:
            print("use 48px")
            textlines = self.run_ocr_48px(img_bbox, list(generate_text_direction(textlines)), self.dictionary, self.model_ocr, 16)
        elif self.use_ctc_model == True:
            print("use ctc")
            textlines = self.run_ocr_ctc(img_bbox, list(generate_text_direction(textlines)), self.dictionary, self.model_ocr, 16)
        else:
            print("use 32px model")
            textlines = self.run_ocr(img_bbox, list(generate_text_direction(textlines)), self.dictionary, self.model_ocr, 16)

        text_regions: List[Quadrilateral] = []
        new_textlines = []
        for (poly_regions, textline_indices, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b) in merge_bboxes_text_region(textlines) :
            text = ''
            logprob_lengths = []
            for textline_idx in textline_indices:
                try:
                    if not text :
                        text = textlines[textline_idx].text
                    else :
                        last_ch = text[-1]
                        cur_ch = textlines[textline_idx].text[0]
                        if ord(last_ch) > 255 and ord(cur_ch) > 255 :
                            text += textlines[textline_idx].text
                        else :
                            if last_ch == '-' and ord(cur_ch) < 255 :
                                text = text[:-1] + textlines[textline_idx].text
                            else :
                                text += ' ' + textlines[textline_idx].text
                except Exception as e:
                    print(e)
                
                logprob_lengths.append((np.log(textlines[textline_idx].prob), len(textlines[textline_idx].text)))
            vc = count_valuable_text(text)
            total_logprobs = 0.0
            for (logprob, length) in logprob_lengths :
                total_logprobs += logprob * length
            total_logprobs /= sum([x[1] for x in logprob_lengths])
            # filter text region without characters
            if vc > 1 :
                region = Quadrilateral(poly_regions, text, np.exp(total_logprobs), fg_r, fg_g, fg_b, bg_r, bg_g, bg_b)
                region.clip(img.shape[1], img.shape[0])
                region.textline_indices = []
                region.majority_dir = majority_dir
                text_regions.append(region)
                for textline_idx in textline_indices :
                    region.textline_indices.append(len(new_textlines))
                    new_textlines.append(textlines[textline_idx])
        textlines: List[Quadrilateral] = new_textlines 
        print(textlines)
        return textlines
        
    def gen_mask(self, img, mask, textlines):
        img_resized, target_ratio, _, pad_w, pad_h = imgproc.resize_aspect_ratio(img, self.img_detect_size, cv2.INTER_LINEAR, mag_ratio = 1)
        mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation = cv2.INTER_LINEAR)
        if pad_h > 0 :
            mask_resized = mask_resized[:-pad_h, :]
        elif pad_w > 0 :
            mask_resized = mask_resized[:, : -pad_w]
        mask_resized = cv2.resize(mask_resized, (img.shape[1] // 2, img.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
        img_resized_2 = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
        mask_resized[mask_resized > 150] = 255
        #cv2.imwrite('mask-resized.png', mask_resized)
        text_lines = [(a.aabb.x // 2, a.aabb.y // 2, a.aabb.w // 2, a.aabb.h // 2) for a in textlines]
        mask_ccs, cc2textline_assignment = filter_masks(mask_resized, text_lines)
        if mask_ccs :
            mask_filtered = reduce(cv2.bitwise_or, mask_ccs)
            final_mask = complete_mask(img_resized_2, mask_ccs, text_lines, cc2textline_assignment)
            final_mask = cv2.resize(final_mask, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LINEAR)
            final_mask[final_mask > 0] = 255
        else :
            final_mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
        return final_mask
        
    def inpaint(self, img, final_mask):
        self.load_inpainting_model()
        img_inpainted, inpaint_input = self.run_inpainting(self.model_inpainting, img, final_mask, self.inpainting_size)
        return img_inpainted
        
    def run_inpainting(self, model_inpainting, img, mask, max_image_size = 1024, pad_size = 4) :
        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]
        height, width, c = img.shape
        if max(img.shape[0: 2]) > max_image_size :
            img = resize_keep_aspect(img, max_image_size)
            mask = resize_keep_aspect(mask, max_image_size)
        h, w, c = img.shape
        if h % pad_size != 0 :
            new_h = (pad_size - (h % pad_size)) + h
        else :
            new_h = h
        if w % pad_size != 0 :
            new_w = (pad_size - (w % pad_size)) + w
        else :
            new_w = w
        if new_h != h or new_w != w :
            img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR_EXACT)
            mask = cv2.resize(mask, (new_w, new_h), interpolation = cv2.INTER_LINEAR_EXACT)
        print(f'Inpainting resolution: {new_w}x{new_h}')
        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1
        if self.use_cuda :
            img_torch = img_torch.cuda()
            mask_torch = mask_torch.cuda()
        with torch.no_grad() :
            img_torch *= (1 - mask_torch)
            img_inpainted_torch = model_inpainting(img_torch, mask_torch)
        img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        if new_h != height or new_w != width :
            img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation = cv2.INTER_LINEAR_EXACT)
        return img_inpainted * mask_original + img_original * (1 - mask_original), (img_torch.cpu() * 127.5 + 127.5).squeeze_(0).permute(1, 2, 0).numpy()
        
if __name__ == '__main__':
    t = Translator()
    
    img=cv2.imread("2.jpg")
    img_bbox = np.copy(img)
    img_bbox = cv2.bilateralFilter(img_bbox, 17, 80, 80)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    textlines, mask=t.detect(img_rgb)
    
    t.ocr(img, img_bbox, textlines)
    cv2.imwrite('mask-org.png', mask)
    mask = t.gen_mask(img_rgb, mask, textlines)
    cv2.imwrite('mask.png', mask)
    inpainted = t.inpaint(img,mask)
    cv2.imwrite('inpainted.png', inpainted)
    
        