#perform binary search

def binsearch(arr, key):

	hi = len(arr)-1;
	lo = 0;

	while(lo <= hi):
		mid = (hi + lo)//2
		s = arr[mid]
		if(key == s): return mid
		elif(key < s): hi = mid - 1
		else: lo = mid + 1
	return -1;

def main():
	arr = list(map(int,input("Enter a list of comma searated values: \n").split(',')))
	key = int(input("Enter element to find: "))
	res = binsearch(arr, key);
	if res == -1:
		print("Element not found!")
	else:
		print("Element found at position ", res)

if(__name__ == "__main__"):
	main()