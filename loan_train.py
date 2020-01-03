import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import test
import copy
import config

def LoanTrain(helper, start_epoch, local_model, target_model, is_poison,state_keys):

    epochs_submit_update_dict = dict()
    num_samples_dict=dict()
    current_number_of_adversaries = len(helper.params['adversary_list'])

    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        last_params_variables = dict()
        client_grad = []  # fg  only works for aggr_epoch_interval=1

        for name, param in target_model.named_parameters():
            last_params_variables[name] = target_model.state_dict()[name].clone()

        state_key = state_keys[model_id]
        ## Synchronize LR and models
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()
        temp_local_epoch=start_epoch-1

        adversarial_index = -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        if is_poison and state_key in helper.params['adversary_list']:
            for adver_index in range(0, len(helper.params['adversary_list'])):
                if state_key == helper.params['adversary_list'][adver_index]:
                    localmodel_poison_epochs = helper.params[str(adver_index) + '_poison_epochs']
                    adversarial_index= adver_index
                    main.logger.info(f'poison local model {state_key} will poison epochs: {localmodel_poison_epochs}')
                    break
            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # attack the global trigger

        trigger_names = []
        trigger_values = []
        if adversarial_index == -1:
            for j in range(0, helper.params['trigger_num']):
                for name in helper.params[str(j) + '_poison_trigger_names']:
                    trigger_names.append(name)
                for value in helper.params[str(j) + '_poison_trigger_values']:
                    trigger_values.append(value)
        else:
            trigger_names = helper.params[str(adversarial_index) + '_poison_trigger_names']
            trigger_values = helper.params[str(adversarial_index) + '_poison_trigger_values']

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):
            ### This is for calculating distances
            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_params_variables[name].clone().detach().requires_grad_(False)

            if is_poison and state_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                main.logger.info('poison_now')
                _, acc_p, _, _ = test.Mytest_poison(helper=helper, epoch=epoch,
                                               model=model, is_poison=True, visualize=False, agent_name_key=state_key)
                main.logger.info(acc_p)
                poison_lr = helper.params['poison_lr']
                if not helper.params['baseline']:
                    if acc_p > 20:
                        poison_lr /= 5
                    if acc_p > 60:
                        poison_lr /= 10

                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']

                poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num],gamma=0.1)
                # acc = acc_initial
                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch+=1
                    poison_data = helper.statehelper_dic[state_key].get_poison_trainloader()
                    if step_lr:
                        scheduler.step()
                        main.logger.info(f'Current lr: {scheduler.get_lr()}')
                    data_iterator = poison_data
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    for batch_id, batch in enumerate(data_iterator):
                        for index in range(0, helper.params['poisoning_per_batch']):
                            if index >= len(batch[1]):
                                break
                            batch[1][index] = helper.params['poison_label_swap']
                            for j in range(0,len(trigger_names)):
                                name= trigger_names[j]
                                value= trigger_values[j]
                                batch[0][index][helper.feature_dict[name]] = value
                            poison_data_count += 1

                        data, targets = helper.statehelper_dic[state_key].get_batch(poison_data, batch, False)
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)
                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)

                        loss = helper.params['alpha_loss'] * class_loss + \
                               (1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward()
                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()
                        poison_optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        '___PoisonTrain {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%), train_poison_data_count{}'.format(model.name, epoch, state_key, internal_epoch,
                                                           total_l, correct, dataset_size,acc, poison_data_count))
                    csv_record.train_result.append(
                        [state_key, temp_local_epoch,
                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])
                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=True,
                                        name=state_key)
                    num_samples_dict[state_key]= dataset_size

                # internal epoch finish
                main.logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
                main.logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                            f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

                ### Adversary wants to scale his weights. Baseline model doesn't do this

                if not helper.params['baseline']:
                    clip_rate = helper.params['scale_weights_poison']
                    main.logger.info(f"Scaling by  {clip_rate}")
                    for key, value in model.state_dict().items():
                        target_value = last_params_variables[key]
                        new_value = target_value + (value - target_value) * clip_rate
                        model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(model, target_params_variables)
                    main.logger.info(
                        f'Scaled Norm after poisoning: '
                        f'{helper.model_global_norm(model)}, distance: {distance}')

                distance = helper.model_dist_norm(model, target_params_variables)
                main.logger.info(f"Total norm for {current_number_of_adversaries} "
                            f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

            else:
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1
                    train_data = helper.statehelper_dic[state_key].get_trainloader()
                    data_iterator = train_data
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    for batch_id, batch in enumerate(data_iterator):
                        optimizer.zero_grad()
                        data, targets = helper.statehelper_dic[state_key].get_batch(data_iterator, batch,
                                                                                    evaluation=False)
                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()

                        # get gradients
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()


                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size

                    main.logger.info(
                        '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, state_key, internal_epoch,
                                                           total_l, correct, dataset_size,
                                                           acc))
                    csv_record.train_result.append([state_key, temp_local_epoch,
                                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=False,
                                        name=state_key)
                    num_samples_dict[state_key] = dataset_size

            # test local model after internal epoch train finish
            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                      model=model, is_poison=False, visualize=True,
                                                                      agent_name_key=state_key)
            csv_record.test_result.append([state_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

            if is_poison:
                if state_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper, epoch=epoch,
                                                                                     model=model, is_poison=True,
                                                                                     visualize=True, agent_name_key=state_key)
                    csv_record.posiontest_result.append([state_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                #  test on local triggers
                if  state_key in helper.params['adversary_list']:
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=state_key + "_combine")

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
                        test.Mytest_poison_agent_trigger(helper=helper, model=model, agent_name_key=state_key)
                    csv_record.poisontriggertest_result.append([state_key, state_key + "_trigger", "", epoch, epoch_loss,
                                                     epoch_acc, epoch_corret,epoch_total])
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=state_key+"_trigger")
            # update the weight and bias
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_params_variables[name])
                last_params_variables[name] = copy.deepcopy(data)

            if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                epochs_local_update_list.append(client_grad)
            else:
                epochs_local_update_list.append(local_model_update_dict)

        epochs_submit_update_dict[state_key] = epochs_local_update_list

    return epochs_submit_update_dict, num_samples_dict
