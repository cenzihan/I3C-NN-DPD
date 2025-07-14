import re


class Calculator:
    def __init__(self):
        self.variables = {}

    def parse_line(self, line):
        if 'out(' in line:
            var_name = line[4:-1]
            if var_name not in self.variables:
                print("<undefined>")
            else:
                value = self.variables[var_name]
                if value == "<underflow>":
                    print("<underflow>")
                elif value == "<overflow>":
                    print("<overflow>")
                else:
                    print(value)
            return
        if not re.match(r"let [_a-zA-Z][_a-zA-Z0-9]* =", line):
            print("<syntax-error>")
            return
        var_name, expr = line[4:].split('=', 1)
        var_name = var_name.strip()
        if not re.match(r"^[_a-zA-Z][_a-zA-Z0-9]*$", var_name):
            print("<syntax-error>")
            return
        tokens = re.findall(r"[_a-zA-Z][_a-zA-Z0-9]*|-?\d+|[+\-*/]", expr)
        stack = []
        for token in tokens:
            if re.match(r"-?\d+", token):
                stack.append(int(token))
            elif re.match(r"[_a-zA-Z][_a-zA-Z0-9]*", token):
                if token not in self.variables:
                    self.variables[var_name] = "<undefined>"
                    return
                else:
                    stack.append(self.variables[token])
            else:
                if len(stack) < 2:
                    self.variables[var_name] = "<syntax-error>"
                    return
                b = stack.pop()
                a = stack.pop()
                if token == "+":
                    res = a + b
                elif token == "-":
                    res = a - b
                elif token == "*":
                    res = a * b
                else:
                    res = a // b
                if res < -2147483648:
                    self.variables[var_name] = "<underflow>"
                    return
                elif res > 2147483647:
                    self.variables[var_name] = "<overflow>"
                    return
                stack.append(res)
        self.variables[var_name] = stack[0]


# Main function to accept inputs and execute the program
if __name__ == "__main__":
    calc = Calculator()
    for _ in range(24):
        try:
            line = input()
            if not line:
                break
            calc.parse_line(line)
        except:
            break