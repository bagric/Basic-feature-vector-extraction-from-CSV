import evm_multi_testing

clo_list = [14, 15]
with open("evm_res_multi_classes_missing.txt", "a") as myfile:
    myfile.write("With missing classes: ")
    for item in clo_list:
        myfile.write(str(item))
        myfile.write(" ")
    myfile.write('\n')
for i in range(1, 5):
    with open("evm_res_multi_classes_missing.txt", "a") as myfile:
        myfile.write("season: ")
        myfile.write(str(i))
        myfile.write('\n')
    evm_multi_testing.main_for_caller(i, 1, 70, 1.0, clo_list)