#to find sum of digits of a number

n = int(input("Enter number to find sum of digits: "))
sum = 0;
n = abs(n)		#for negative numbers
while(n):
	sum += n % 10;
	n = n // 10;

print("Sum is: ", sum)