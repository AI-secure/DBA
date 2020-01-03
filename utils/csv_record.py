
import csv
import copy
train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy", "correct_data",
                    "total_data"]
test_fileHeader = ["model", "epoch", "average_loss", "accuracy", "correct_data", "total_data"]
train_result = []  # train_fileHeader
test_result = []  # test_fileHeader
posiontest_result = []  # test_fileHeader

triggertest_fileHeader = ["model", "trigger_name", "trigger_value", "epoch", "average_loss", "accuracy", "correct_data",
                          "total_data"]
poisontriggertest_result = []  # triggertest_fileHeader

posion_test_result = []  # train_fileHeader
posion_posiontest_result = []  # train_fileHeader
weight_result=[]
scale_result=[]
scale_temp_one_row=[]

def save_result_csv(epoch, is_posion,folder_path):
    train_csvFile = open(f'{folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csvFile)
    train_writer.writerow(train_fileHeader)
    train_writer.writerows(train_result)
    train_csvFile.close()

    test_csvFile = open(f'{folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_result)
    test_csvFile.close()

    if len(weight_result)>0:
        weight_csvFile=  open(f'{folder_path}/weight_result.csv', "w")
        weight_writer = csv.writer(weight_csvFile)
        weight_writer.writerows(weight_result)
        weight_csvFile.close()

    if len(scale_temp_one_row)>0:
        _csvFile=  open(f'{folder_path}/scale_result.csv', "w")
        _writer = csv.writer(_csvFile)
        scale_result.append(copy.deepcopy(scale_temp_one_row))
        scale_temp_one_row.clear()
        _writer.writerows(scale_result)
        _csvFile.close()

    if is_posion:
        test_csvFile = open(f'{folder_path}/posiontest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()

        test_csvFile = open(f'{folder_path}/poisontriggertest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(triggertest_fileHeader)
        test_writer.writerows(poisontriggertest_result)
        test_csvFile.close()

def add_weight_result(name,weight,alpha):
    weight_result.append(name)
    weight_result.append(weight)
    weight_result.append(alpha)


