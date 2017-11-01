from collections import deque

def search(lines, pattern, history):
    previous_lines = deque(maxlen=history)
    for line in lines:
        if pattern in line:
            yield line, previous_lines
        previous_lines.append(line)

if __name__ == '__main__':
    with open('somefile.txt') as f:
        for line, prevlines in search(f, 'python', 5):
            if len(prevlines)>4:
                for pline in prevlines:
                    print(pline, end = '')
            print(line, end='')
            print('-'*20)
