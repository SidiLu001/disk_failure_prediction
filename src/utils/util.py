import os

getBasename = lambda path: os.path.splitext(os.path.split(path)[1])[0]

if __name__ == '__main__':
    print(getBasename(__file__))