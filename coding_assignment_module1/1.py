#make a list and tuple from comma separated values given by the user

x_list = list(map(int,input("Enter a list of comma searated values: \n").split(',')))
x_tuple = tuple(x_list)
print(x_list, x_tuple, sep = " ")