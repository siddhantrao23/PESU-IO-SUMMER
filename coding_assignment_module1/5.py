try:
	num = int(input("Enter a string: "))
except Exception as e:
	print("String is not numeric!")
else:
	print("String is numeric")
