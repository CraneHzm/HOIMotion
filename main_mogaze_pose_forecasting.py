from utils import mogaze_dataset, seed_torch
from model import pose_forecasting
from utils.opt import options
from utils.weight_init import weight_init
from utils import log
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import torch.optim as optim
import os
import math

def main(opt):
    # set the random seed to ensure reproducibility
    seed_torch.seed_torch(seed=0)
    torch.set_num_threads(1)
    
    learning_rate = opt.learning_rate
    print('>>> create model')
    net_pred = pose_forecasting.pose_forecasting(opt=opt).to(opt.cuda_idx)
    net_pred.apply(weight_init)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.learning_rate)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    print('>>> loading datasets')
    
    data_dir = opt.data_dir
    input_n = opt.input_n
    output_n = opt.output_n
    test_participant = opt.test_id
    actions = opt.actions
    sample_rate = opt.train_sample_rate
    if test_participant == 1:
        train_subjects = ['p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p1_1', 'p1_2']
    if test_participant == 2:
        train_subjects = ['p1_1', 'p1_2', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p2_1']
    if test_participant == 4:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p4_1']
    if test_participant == 5:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p5_1']
    if test_participant == 6:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p7_1', 'p7_3']
        test_subjects = ['p6_1', 'p6_2']
    if test_participant == 7:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2']
        test_subjects = ['p7_1', 'p7_3']
        
    train_dataset = mogaze_dataset.mogaze_dataset(data_dir, train_subjects, input_n, output_n, actions, sample_rate)
    train_data_size = train_dataset.pose_head_objects.shape
    print("Training data size: {}".format(train_data_size))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_dataset = mogaze_dataset.mogaze_dataset(data_dir, test_subjects, input_n, output_n, actions)
    valid_data_size = valid_dataset.pose_head_objects.shape
    print("Validation data size: {}".format(valid_data_size))
    valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # training
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTraining starts at ' + local_time)
    start_time = datetime.datetime.now()
    start_epoch = 1
    
    err_best = 1000
    best_epoch = 0
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma, last_epoch=-1)
    for epo in range(start_epoch, opt.epoch + 1):
        is_best = False            
        learning_rate = exp_lr.optimizer.param_groups[0]["lr"]
        
        train_start_time = datetime.datetime.now()
        result_train = run_model(net_pred, optimizer, is_train=1, data_loader=train_loader, opt=opt)
        train_end_time = datetime.datetime.now()
        train_time = (train_end_time - train_start_time).seconds*1000
        train_batch_num = math.ceil(train_data_size[0]/opt.batch_size)
        train_time_per_batch = math.ceil(train_time/train_batch_num)
        print('\nTraining time per batch: {} ms'.format(train_time_per_batch))
        
        exp_lr.step()
        rng_state = torch.get_rng_state()
        if epo % opt.validation_epoch == 0:
            print("\nTest participant: {}\n".format(test_participant))
            print('>>> training epoch: {:d}, lr: {:.12f}'.format(epo, learning_rate))
            print('Training data size: {}'.format(train_data_size))
            print('Average training error: {:.1f} mm'.format(result_train['prediction_error_average']*1000))
            
            test_start_time = datetime.datetime.now()
            result_valid = run_model(net_pred, is_train=0, data_loader=valid_loader, opt=opt)
            test_end_time = datetime.datetime.now()
            test_time = (test_end_time - test_start_time).seconds*1000
            test_batch_num = math.ceil(valid_data_size[0]/opt.test_batch_size)
            test_time_per_batch = math.ceil(test_time/test_batch_num)
            print('\nTest time per batch: {} ms'.format(test_time_per_batch))
            print('Validation data size: {}'.format(valid_data_size))
            
            validation_times = np.arange(1, 11)*100               
            print('Average baseline error: {:.1f} mm'.format(result_valid['baseline_error_average']*1000))
            for validation_time in validation_times:
                key = "baseline_" + str(validation_time) + "_ms"
                print('{}: {:.1f} mm'.format(key, result_valid[key]*1000))
                
            print('Average validation error: {:.1f} mm'.format(result_valid['prediction_error_average']*1000))
            for validation_time in validation_times:
                key = "prediction_" + str(validation_time) + "_ms"
                print('{}: {:.1f} mm'.format(key, result_valid[key]*1000))                                                                            
            if result_valid['prediction_error_average'] < err_best:
                err_best = result_valid['prediction_error_average']
                is_best = True
                best_epoch = epo
                
            print('Best validation error: {:.1f} mm, best epoch: {}'.format(err_best*1000, best_epoch))                                                
            end_time = datetime.datetime.now()
            total_training_time = (end_time - start_time).seconds/60
            print('\nTotal training time: {:.1f} min'.format(total_training_time))
            local_time = time.asctime(time.localtime(time.time()))
            print('\nTraining ends at ' + local_time)
            
            # save the results
            head = np.array(['epoch', 'lr', 'training_error'])
            training_error = format(result_train['prediction_error_average']*1000, ".1f")
            ret_log = np.array([epo, learning_rate, training_error])
                        
            validation_times = np.arange(1, 11)*100
            for validation_time in validation_times:                
                key = "prediction_" + str(validation_time) + "_ms"
                head = np.append(head, key)
                value = format(result_valid[key]*1000, ".1f")
                ret_log = np.append(ret_log, value)
            prediction_error = format(result_valid['prediction_error_average']*1000, ".1f")
            head = np.append(head, 'prediction_error_average')
            ret_log = np.append(ret_log, prediction_error)
            
            for validation_time in validation_times:
                key = "baseline_" + str(validation_time) + "_ms"
                head = np.append(head, key)
                value = format(result_valid[key]*1000, ".1f")
                ret_log = np.append(ret_log, value)
            baseline_error = format(result_valid['baseline_error_average']*1000, ".1f")                       
            head = np.append(head, 'baseline_error_average')
            ret_log = np.append(ret_log, baseline_error)
            
            csv_name = 'pose_forecasting_results'
            model_name = 'pose_forecasting_model.pt'
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1), file_name=csv_name)
            
            # save the model
            log.save_ckpt({'epoch': epo,
                           'lr': learning_rate,
                           'err': result_valid['prediction_error_average'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                            opt=opt,
                            file_name = model_name)
                          
        torch.set_rng_state(rng_state)

        
def eval(opt):
    print('>>> create model')
    net_pred = pose_forecasting.pose_forecasting(opt=opt).to(opt.cuda_idx)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    
    model_name = 'pose_forecasting_model.pt'
    model_path = os.path.join(opt.ckpt, model_name)
    print(">>> loading ckpt from '{}'".format(model_path))
    ckpt = torch.load(model_path)
    net_pred.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
                
    print('>>> loading datasets')
    data_dir = opt.data_dir
    input_n = opt.input_n
    output_n = opt.output_n
    actions = opt.actions
    test_participant = opt.test_id
    if test_participant == 1:        
        train_subjects = ['p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p1_1', 'p1_2']            
    if test_participant == 2:            
        train_subjects = ['p1_1', 'p1_2', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p2_1']
    if test_participant == 4:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p4_1']
    if test_participant == 5:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p6_1', 'p6_2', 'p7_1','p7_3']
        test_subjects = ['p5_1']
    if test_participant == 6:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p7_1', 'p7_3']
        test_subjects = ['p6_1', 'p6_2']
    if test_participant == 7:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2']
        test_subjects = ['p7_1', 'p7_3']
        
    test_dataset = mogaze_dataset.mogaze_dataset(data_dir, test_subjects, input_n, output_n, actions)
    test_data_size = test_dataset.pose_head_objects.shape
    print("Test data size: {}".format(test_data_size))                
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # test
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest starts at ' + local_time)
    start_time = datetime.datetime.now()
    print("\nAction: {}, Test participant: {}\n".format(actions, test_participant))
    if opt.save_predictions:
        ret_test, predictions = run_model(net_pred, is_train=0, data_loader=test_loader, opt=opt)
    else:
        ret_test = run_model(net_pred, is_train=0, data_loader=test_loader, opt=opt)

    if opt.test_joint_error:
        test_times = np.arange(1, opt.joint_number + 1)        
    else:    
        test_times = np.arange(1, 11)*100
        
    print('Average baseline error: {:.1f} mm'.format(ret_test['baseline_error_average']*1000))
    for test_time in test_times:
        if opt.test_joint_error:
            key = "baseline_" + str(test_time) + "_joint"            
        else:
            key = "baseline_" + str(test_time) + "_ms"
        print('{}: {:.1f} mm'.format(key, ret_test[key]*1000))
        
    print('Average prediction error: {:.1f} mm'.format(ret_test['prediction_error_average']*1000))
    for test_time in test_times:
        if opt.test_joint_error:
            key = "prediction_" + str(test_time) + "_joint"            
        else:
            key = "prediction_" + str(test_time) + "_ms"
        print('{}: {:.1f} mm'.format(key, ret_test[key]*1000))                                           
    end_time = datetime.datetime.now()
    total_test_time = (end_time - start_time).seconds/60
    print('\nTotal test time: {:.1f} min'.format(total_test_time))  
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest ends at ' + local_time)
    
    # save the results
    head = []
    ret_log = []                              
    for test_time in test_times:
        if opt.test_joint_error:
            key = "prediction_" + str(test_time) + "_joint"            
        else:
            key = "prediction_" + str(test_time) + "_ms"
        head = np.append(head, key)
        value = format(ret_test[key]*1000, ".1f")
        ret_log = np.append(ret_log, value)
    prediction_error = format(ret_test['prediction_error_average']*1000, ".1f")
    head = np.append(head, 'prediction_error_average')
    ret_log = np.append(ret_log, prediction_error)
    
    for test_time in test_times:
        if opt.test_joint_error:
            key = "baseline_" + str(test_time) + "_joint"            
        else:
            key = "baseline_" + str(test_time) + "_ms"
        head = np.append(head, key)
        value = format(ret_test[key]*1000, ".1f")
        ret_log = np.append(ret_log, value)
    baseline_error = format(ret_test['baseline_error_average']*1000, ".1f")                       
    head = np.append(head, 'baseline_error_average')
    ret_log = np.append(ret_log, baseline_error)
    
    csv_name = 'pose_forecasting_results_test'
    if opt.test_joint_error:
        csv_name = 'pose_forecasting_results_test_joint'        
            
    log.save_csv_log(opt, head, ret_log, is_create=1, file_name=csv_name)
    
    if opt.save_predictions:
        ground_truth = test_dataset[:, opt.input_n:, :opt.joint_number*3]
        
        ground_truth1 = ground_truth.reshape([-1, output_n, opt.joint_number, 3])
        predictions1 = predictions.reshape([-1, output_n, opt.joint_number, 3])               
        prediction_errors = np.mean(np.mean(np.linalg.norm(ground_truth1 - predictions1, axis=3), axis=2), axis=1)
        print('Average prediction error: {:.1f} mm'.format(np.mean(prediction_errors)*1000))
        
        ground_truth_path = os.path.join(opt.ckpt, "ground_truth.npy")
        np.save(ground_truth_path, ground_truth)
        predictions_path = os.path.join(opt.ckpt, "predictions.npy")
        np.save(predictions_path, predictions)
        prediction_errors_path = os.path.join(opt.ckpt, "prediction_errors.npy")
        np.save(prediction_errors_path, prediction_errors)
        
        
def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

    
def run_model(net_pred, optimizer=None, is_train=1, data_loader=None, opt=None):
    if is_train == 1:
        net_pred.train()
    else:
        net_pred.eval()
    
    if opt.is_eval and opt.save_predictions:
        predictions = []
        
    prediction_error_average = 0
    if is_train == 0:
        baseline_error_average = 0        
        fps = 30.0
        if option.is_eval and opt.test_joint_error:
            titles = np.arange(1, 1 + opt.joint_number)
            prediction_errors = np.zeros([opt.joint_number])
            baseline_errors = np.zeros([opt.joint_number])
        else:
            titles = ((np.array(range(opt.output_n)) + 1)*1000/fps).astype(int)                
            prediction_errors = np.zeros([opt.output_n])
            baseline_errors = np.zeros([opt.output_n])
            
    n = 0
    input_n = opt.input_n
    output_n = opt.output_n
    
    for i, (data) in enumerate(data_loader):
        batch_size, seq_n, dim = data.shape
        joint_number = opt.joint_number
        # when only one sample in this batch
        if batch_size == 1 and is_train == 1:
            continue
        n += batch_size
        data = data.float().to(opt.cuda_idx)
        
        pose_ground_truth = data.clone()[:, input_n:, :joint_number*3].reshape([-1, output_n, joint_number, 3])        
        pose_input = data.clone()[:, :input_n, :joint_number*3]
        head_input = data.clone()[:, :input_n, joint_number*3:joint_number*3+3]
        
        object_point_num = 8 # 8 points in a bounding box
        dynamic_num_all = 2
        static_num_all = 2
        object_num_dynamic = opt.object_num_dynamic
        objects_dynamic = data.clone()[:, :input_n, joint_number*3+3:joint_number*3+3+object_point_num*3*object_num_dynamic]                
        object_num_static = opt.object_num_static
        objects_static = data.clone()[:, :input_n, joint_number*3+3+dynamic_num_all*3*object_point_num:joint_number*3+3+dynamic_num_all*3*object_point_num+3*object_point_num*object_num_static]
                                           
        input = pose_input
        input = torch.cat((input, head_input), dim=2)
        input = torch.cat((input, objects_dynamic), dim=2)        
        input = torch.cat((input, objects_static), dim=2)        
        pose_prediction = net_pred(input, input_n=input_n, output_n=output_n)
        
        if opt.is_eval and opt.save_predictions:
            pose_prediction_cpu = pose_prediction.cpu().data.numpy()
            if len(predictions) == 0:
                predictions = pose_prediction_cpu                
            else:
                predictions = np.concatenate((predictions, pose_prediction_cpu), axis=0)                           
        pose_prediction = pose_prediction.reshape([-1, output_n, joint_number, 3])        
        loss = torch.mean(torch.norm(pose_ground_truth - pose_prediction, dim=3))
        
        # use velocity loss
        if opt.velocity_loss == 1:
            pose_ground_truth_vel = gen_velocity(pose_ground_truth)
            prediction_vel = gen_velocity(pose_prediction)
            loss_vel = torch.mean(torch.norm(pose_ground_truth_vel - prediction_vel, dim=3))
            loss += loss_vel
            
        if is_train == 1:
            optimizer.zero_grad()
            loss.backward()                        
            optimizer.step()
            
        # Calculate prediction errors
        mpjpe_data = torch.mean(torch.norm(pose_ground_truth - pose_prediction, dim=3))
        prediction_error_average += mpjpe_data.cpu().data.numpy() * batch_size
        if is_train == 0:
            # zero velocity baseline, i.e. repeat last frame as the predictions
            pose_baseline = pose_input[:, -1:, :].expand(-1, output_n, -1).clone().reshape([-1, output_n, joint_number, 3])
            mpjpe_data = torch.mean(torch.norm(pose_ground_truth - pose_baseline, dim=3))
            baseline_error_average += mpjpe_data.cpu().data.numpy() * batch_size
                        
            if option.is_eval and opt.test_joint_error:
                mpjpe_data = torch.sum(torch.mean(torch.norm(pose_ground_truth - pose_prediction, dim=3), dim=1), dim=0)            
            else:    
                mpjpe_data = torch.sum(torch.mean(torch.norm(pose_ground_truth - pose_prediction, dim=3), dim=2), dim=0)                
            prediction_errors += mpjpe_data.cpu().data.numpy()
            
            if option.is_eval and opt.test_joint_error:
                mpjpe_data = torch.sum(torch.mean(torch.norm(pose_ground_truth - pose_baseline, dim=3), dim=1), dim=0)
            else:
                mpjpe_data = torch.sum(torch.mean(torch.norm(pose_ground_truth - pose_baseline, dim=3), dim=2), dim=0)               
            baseline_errors += mpjpe_data.cpu().data.numpy()
                
    ret = {}
    if is_train == 1:
        ret["prediction_error_average"] = prediction_error_average / n
    else:
        ret["prediction_error_average"] = prediction_error_average / n
        ret["baseline_error_average"] = baseline_error_average / n
        prediction_errors = prediction_errors / n
        baseline_errors = baseline_errors / n        
        
        if option.is_eval and opt.test_joint_error:
            for j in range(opt.joint_number):      
                ret["baseline_{:d}_joint".format(titles[j])] = baseline_errors[j]
                ret["prediction_{:d}_joint".format(titles[j])] = prediction_errors[j]                              
        else:
            for j in range(output_n):      
                ret["baseline_{:d}_ms".format(titles[j])] = baseline_errors[j]
                ret["prediction_{:d}_ms".format(titles[j])] = prediction_errors[j]                
                
    if opt.is_eval and opt.save_predictions:        
        return ret, predictions
    else:
        return ret

        
if __name__ == '__main__':    
    option = options().parse()
    if option.is_eval == False:
        main(option)
    else:
        eval(option)