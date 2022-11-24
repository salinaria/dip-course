def func(seed,dims):
    arr = []
    for i in range(dims[0]):
        arr.append([])
        for j in range(dims[1]):
            if i == 0:
                arr[0].append(seed)
            elif j == 0:
                arr[i].append(seed)
            else:
                a = arr[i][j-1] + arr[i-1][j-1] + arr[i-1][j] 
                arr[i].append(a)
    print(seed,dims)
    print()  
    for i in arr:
        for j in i:
            print(j,end='  ')
        print()

func(3,(5,6))