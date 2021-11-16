import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint,Refine
from models.afwm import AFWM
from models.unet2 import UNet,UNet1,UNet2
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import ipdb
from torch.autograd import Variable
from util import util
#from pylab import *
from sklearn.ensemble import IsolationForest
from PIL import Image
NC=4
def gen_noise(shape):
        noise = np.zeros(shape, dtype=np.uint8)
        ### noise
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()
def morpho(mask,iter,bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new=[]
    for i in range(len(mask)):
        tem=mask[i].cpu().detach().numpy().squeeze().reshape(256,192,1)*255
        tem=tem.astype(np.uint8)
        if bigger:
            tem=cv2.dilate(tem,kernel,iterations=iter)
        else:
            tem=cv2.erode(tem,kernel,iterations=iter)
        tem=tem.astype(np.float64)
        tem=tem.reshape(1,256,192)
        new.append(tem.astype(np.float64)/255.0)
    new=np.stack(new)
    new=torch.FloatTensor(new).cuda()
    return new
def encode_input(label_map, nc = NC):
    size = label_map.size()
    oneHot_size = (size[0], nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    #ipdb.set_trace()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    #ipdb.set_trace()

    input_label = Variable(input_label)

    return input_label
def generate_label_plain(inputs, nc = NC):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, nc, 256,192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256,192)

    return label_batch
def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label

def ger_average_color(mask,arms):
    color=torch.zeros(arms.shape).cuda()
    for i in range(arms.shape[0]):
        count = len(torch.nonzero(mask[i, :, :, :]))
        if count < 10:
            color[i, 0, :, :]=0
            color[i, 1, :, :]=0
            color[i, 2, :, :]=0

        else:
            color[i,0,:,:]=arms[i,0,:,:].sum()/count
            color[i,1,:,:]=arms[i,1,:,:].sum()/count
            color[i,2,:,:]=arms[i,2,:,:].sum()/count
    return color
opt = TestOptions().parse()

start_epoch, epoch_iter = 1, 0
creationL1 = nn.L1Loss()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)


warp_model = AFWM(opt, 3)
print(warp_model)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, '/PF-AFN-main_XP/PF-AFN_test/checkpoints/PFAFN_warp_epoch.pth')

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
print(gen_model)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model,  '/PF-AFN-main_XP/PF-AFN_test/checkpoints/PFAFN_gen_epoch.pth')

comp_mask_model = UNet1(in_channels=22)
load_checkpoint(comp_mask_model , '/PF-AFN-main_XP/PF-AFN_test/checkpoints/comp_mask_epoch.pth')
comp_mask_model .eval()
comp_mask_model .cuda()

comp_occu_model = UNet2(in_channels=22)
load_checkpoint(comp_occu_model , '/PF-AFN-main_XP/PF-AFN_test/checkpoints/comp_sleeve_epoch.pth')
comp_occu_model.eval()
comp_occu_model.cuda()


#refine_model = ResUnetGenerator(6, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
#load_checkpoint(refine_model, '/home/yzj6850/cjy/mixup/checkpoints_v2/flow/comp_occu_model_epoch_021.pth')

#refine_model.eval()
#refine_model.cuda()





total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

for epoch in range(1,2):
    l1_loss = 0
    for i, data in enumerate(dataset, start=epoch_iter):

        real_image = data['image']
        clothes = data['clothes']
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        pose = data['pose']
        ##edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = clothes * edge        
        pose = data['pose']
        #image_test = data['image_test']
        keep_label = torch.FloatTensor((data['label'].cpu().numpy()==11).astype(np.int) + (data['label'].cpu().numpy()==13).astype(np.int)).cuda()

        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy()==4).astype(np.int))
        shape = person_clothes_edge.size()
        face = torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int))
        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')
        person_arm_img = keep_label * real_image.cuda()
        warped_edge_un = torch.FloatTensor((warped_edge.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()

        warped_edge = torch.FloatTensor((warped_edge.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()

        #image_test_cloth = image_test.cuda() * warped_edge


        occu_skin_mask = person_clothes_edge.cuda() * (1 - warped_edge_un)
        arm_label = torch.FloatTensor((data['label'].cpu().numpy()==11).astype(np.float) + (data['label'].cpu().numpy()==13).astype(np.float)*2).cuda()
        keep_skin_mask = arm_label * (1-warped_edge_un)
        keep_arm_label = (data['label'].cpu().numpy()==11).astype(np.float) + (data['label'].cpu().numpy()==13).astype(np.float)

        arm_image = keep_arm_label * real_image.detach().cpu().numpy()

        keep_arm_label = torch.FloatTensor(keep_arm_label).cuda()

        occu_w_arm_label = keep_skin_mask + occu_skin_mask*3
        encode_occu_w_arm_label = encode_input(occu_w_arm_label)
        gen_arm_mask = comp_occu_model(torch.cat([pose.cuda(), encode_occu_w_arm_label],1))
        gen_arm_mask = torch.sigmoid(gen_arm_mask)
        ground, gen_arm_1, gen_arm_2 = torch.split(gen_arm_mask,[1,1,1],1) 
        gen_arm_1 =  torch.FloatTensor(((gen_arm_1.detach().cpu().numpy() > 0.5)).astype(np.float)).cuda()
        gen_arm_2 =  torch.FloatTensor(((gen_arm_2.detach().cpu().numpy() > 0.5)).astype(np.float)).cuda() 
        #clo_dilate = warped_edge_un * keep_arm_label
        clo_dilate = morpho(warped_edge_un,  8)
        clo_dilate = torch.FloatTensor((clo_dilate.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
        gen_arm_1_dilate = gen_arm_1*(1-clo_dilate)
        gen_arm_2_dilate = gen_arm_2*(1-clo_dilate)
        gen_arm_1_dilate = morpho(gen_arm_1_dilate,30)
        gen_arm_2_dilate = morpho(gen_arm_2_dilate,30)
        gen_arm_1_dilate =  torch.FloatTensor(((gen_arm_1_dilate.detach().cpu().numpy() > 0.5)).astype(np.float)).cuda()
        gen_arm_2_dilate =  torch.FloatTensor(((gen_arm_2_dilate.detach().cpu().numpy() > 0.5)).astype(np.float)).cuda() #本来0.3

        gen_arm_1 = gen_arm_1*gen_arm_1_dilate
        gen_arm_2 = gen_arm_2*gen_arm_2_dilate
 
        gen_arm_1 = gen_arm_1* (1-warped_edge_un)  
        gen_arm_2 = gen_arm_2* (1-warped_edge_un)     
          
        deoccu_w_arm_label = gen_arm_1*1 + gen_arm_2*2 + warped_edge_un*3
        encode_deoccu_w_arm_label = encode_input(deoccu_w_arm_label)
        deoccu_arm_mask = comp_mask_model (torch.cat([pose.cuda(), encode_deoccu_w_arm_label],1))
        deoccu_arm_mask = torch.sigmoid(deoccu_arm_mask)
        ground_deoccu, arm_1, arm_2 = torch.split(deoccu_arm_mask,[1,1,1],1) 

        arm_1 =  torch.FloatTensor(((arm_1.detach().cpu().numpy() >0.5)).astype(np.float)).cuda()
        arm_2 =  torch.FloatTensor(((arm_2.detach().cpu().numpy() >0.5)).astype(np.float)).cuda()

        warped_edge_un[arm_1==1]=0
        warped_edge_un[arm_2==1]=0
        gen_arm_label = arm_1 + arm_2 - (arm_1 * arm_2)
        warped_cloth_un = warped_cloth * warped_edge_un

        img_deal = real_image.cuda() * (1 - person_clothes_edge.cuda()) * (1 - warped_edge_un)


        gen_inputs = torch.cat([real_image.cuda(), warped_cloth_un, warped_edge_un], 1)
        #gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge_un
        p_tryon = warped_cloth_un * m_composite + p_rendered * (1 - m_composite)
        
        #warped_cloth = p_tryon * warped_edge
        p_arm = p_tryon.cuda() * gen_arm_label
        #image_test_arm = image_test.cuda() * gen_arm_label
        #p_tryon = p_tryon * (1 - keep_label) + real_image.cuda() * keep_label

        #skin_color = ger_average_color(face, face * real_image)
        #gen_inputs = torch.cat([p_tryon, skin_color],1)
        #gen_outputs = refine_model(gen_inputs)  
        #p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        #p_rendered = torch.tanh(p_rendered)
        #m_composite = torch.sigmoid(m_composite)

        #m_composite = m_composite * gen_arm_label
        #p_tryon1 = p_tryon.cuda() * (1 - m_composite) + p_rendered * m_composite
        #gen_inputs = torch.cat([p_tryon, keep_left_arm],1)
        #gen_outputs= refine_model(gen_inputs)
        #p_rendered, m_composite = torch.split(gen_outputs,[3,1],1)
        #p_rendered = torch.tanh(p_rendered)
        #m_composite = torch.sigmoid(m_composite)
        #m_composite = m_composite * keep_arm_label
        #p_tryon =p_tryon * (1 - m_composite) + p_rendered * m_composite

        path = 'results/' + opt.name
        os.makedirs(path, exist_ok=True)
        sub8_path = 'P_person'
        sub_path = 'p_gen_un_cloth_W_do/'
        #sub1_path = 'P_gen_arm/'
        sub1_path = 'P_enhance_De_occu_test/'
        sub2_path = 'P_cloth_de_occu_test/'
        sub3_path = 'P_cloth_init_PFAFN_test/'
        sub4_path = 'P_cloth_test/'
        #sub4_path = 'P_reference_arm/'
        sub5_path = 'P_complex_5/'
        sub6_path = 'P_cloth_init/'
        sub7_path = 'P_gen_cloth_init_PFAFN/'
        os.makedirs(sub_path,exist_ok=True)
        os.makedirs(sub1_path,exist_ok=True)
        os.makedirs(sub2_path,exist_ok=True)
        os.makedirs(sub3_path,exist_ok=True)
        os.makedirs(sub4_path,exist_ok=True)
        os.makedirs(sub5_path,exist_ok=True)
        os.makedirs(sub6_path,exist_ok=True)
        os.makedirs(sub7_path,exist_ok=True)
        os.makedirs(sub8_path,exist_ok=True)
        #ipdb.set_trace()
        if step % 1 == 0:
            o = img_deal.float().cuda()
            a = real_image.float().cuda()
            b = person_arm_img.float().cuda()
            c = p_tryon.float().cuda()
            d = clothes.float().cuda()
            l = warped_cloth_un.float().cuda()
            f = p_arm.float().cuda()
            g = warped_edge
            #d = image_test_arm
            e = person_arm_img.float().cuda()
            #d = p_tryon1.float().cuda()
            #d = generate_label_color(arm_label).float().cuda()
            #e = generate_label_color(generate_label_plain(encode_occu_w_arm_label)).float().cuda()
            #f = generate_label_color(generate_label_plain(gen_arm_mask,3)).float().cuda()
            #g = generate_label_color(generate_label_plain(encode_deoccu_w_arm_label)).float().cuda()
            #h = generate_label_color(generate_label_plain(deoccu_arm_mask ,3)).float().cuda()
            k = p_rendered.float().cuda()
            o = warped_cloth.float().cuda()
            #l = generate_label_color(need_dilate).float().cuda()
            #combine = torch.cat([o[0],a[0],b[0],l[0],c[0],d[0],e[0],f[0],g[0],h[0],k[0]], 2).squeeze()
            #combine = torch.cat([o[0],a[0],b[0],l[0],c[0],d[0],k[0]], 2).squeeze()
            #cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            #rgb=(cv_img*255).astype(np.uint8)
            #bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub_path+'/'+str(step)+'.jpg',bgr)
       
            combine = o[0]
            #combine = torch.cat([a[0],c[0]], 1).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub3_path+'/'+str(step)+'.jpg',bgr)  

            combine = c[0]
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(sub8_path+'/'+str(step)+'.jpg',bgr)  

            combine = l[0]
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub2_path+'/'+str(step)+'.jpg',bgr)  

            combine = torch.cat([a[0],d[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub4_path+'/'+str(step)+'.jpg',bgr)  

            combine = o[0]
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub7_path+'/'+str(step)+'.jpg',bgr)  


            combine = l[0]
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub_path+'/'+str(step)+'.jpg',bgr)  

            combine = b[0]
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub4_path+'/'+data['name'][0],bgr)  

            #combine = d[0]
            #cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub3_path+'/'+str(step)+'.jpg',bgr)  

            #combine = d[0]
            #cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            #rgb=(cv_img*255).astype(np.uint8)
            #bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sub2_path+'/'+str(step)+'.jpg',bgr)   

        step += 1
        if epoch_iter >= dataset_size:
            break



