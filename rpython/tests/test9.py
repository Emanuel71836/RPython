def heavy(n: int) -> int:
    i = 0
    res = 0
    while i < n:
        res = res + i
        i = i + 1
    return res

def main():
    i = 0
    res = 0
    while i < 20000:
        res = heavy(5000)
        i = i + 1
    print(res)

if __name__ == "__main__":
    main()