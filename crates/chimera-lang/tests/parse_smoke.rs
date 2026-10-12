use chimera_core::SourceFile;
use chimera_lang::parser::parse;

fn check(name: &str, src: &str) -> bool {
    let (stmts, errs) = parse(src, 0);
    if errs.is_empty() {
        println!("ok   {name}: {} stmts", stmts.len());
        true
    } else {
        let file = SourceFile::new(name, src);
        println!("FAIL {name}");
        for e in &errs {
            println!("{}", e.render(&file, false));
        }
        false
    }
}

#[test]
fn dialect_smoke() {
    let cases: &[(&str, &str)] = &[
        (
            "python",
            "def add(a, b):\n    return a + b\n\nfor i in range(3):\n    if i % 2 == 0:\n        print(i)\n    elif i == 1:\n        print('one')\n    else:\n        pass\nprint(add(1, 2))\n",
        ),
        (
            "js",
            "function add(a, b) {\n  return a + b;\n}\nconst xs = [1, 2, 3].map(x => x * 2);\nfor (let i = 0; i < 3; i++) {\n  console.log(i);\n}\nif (xs.length > 2) { console.log(\"big\") } else { console.log(\"small\") }\n",
        ),
        (
            "ruby",
            "def greet(name)\n  puts \"Hello, #{name}!\"\nend\n[1, 2, 3].each do |x|\n  puts x * 2\nend\nputs \"done\" if true\nclass Dog < Animal\n  def initialize(name)\n    @name = name\n  end\n  def speak\n    \"#{@name} barks\"\n  end\nend\n",
        ),
        (
            "lua",
            "local function fib(n)\n  if n < 2 then\n    return n\n  end\n  return fib(n - 1) + fib(n - 2)\nend\nfor i = 1, 10 do\n  print(fib(i))\nend\nrepeat\n  x = x - 1\nuntil x == 0\n",
        ),
        (
            "rust",
            "fn main() {\n    let mut total = 0;\n    for i in 0..10 {\n        total += i;\n    }\n    println!(\"{}\", total);\n    let v = vec![1, 2, 3];\n    match total {\n        0 => println!(\"zero\"),\n        n if n > 10 => println!(\"big {}\", n),\n        _ => println!(\"other\"),\n    }\n}\nstruct Point { x: f64, y: f64 }\nimpl Point {\n    fn norm(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }\n}\n",
        ),
        (
            "c",
            "#include <stdio.h>\nint square(int x) {\n    return x * x;\n}\nint main() {\n    int total = 0;\n    for (int i = 0; i < 5; i++) {\n        total += square(i);\n    }\n    printf(\"%d\\n\", total);\n    return 0;\n}\n",
        ),
        (
            "java",
            "public class Main {\n    public static void main(String[] args) {\n        int x = 5;\n        System.out.println(\"x = \" + x);\n    }\n}\n",
        ),
        (
            "go",
            "package main\nimport \"fmt\"\nfunc main() {\n    sum := 0\n    for i := 0; i < 10; i++ {\n        sum += i\n    }\n    fmt.Println(sum)\n}\ntype Point struct {\n    X int\n    Y int\n}\nfunc (p Point) Sum() int {\n    return p.X + p.Y\n}\n",
        ),
        (
            "haskell",
            "fib 0 = 0\nfib 1 = 1\nfib n = fib (n - 1) + fib (n - 2)\nsquares = [x * x | x <- [1..10], even x]\nabs' n | n < 0 = -n | otherwise = n\n",
        ),
        (
            "ocaml",
            "let rec fact n = if n <= 1 then 1 else n * fact (n - 1)\nlet () = print_endline (string_of_int (fact 5))\ntype shape = Circle of float | Square of float\nlet area s = match s with\n  | Circle r -> 3.14 *. r *. r\n  | Square w -> w *. w\n",
        ),
        (
            "english",
            "set total to 0\nadd 5 to total\nrepeat 3 times:\n    increase total by 2\nif total is greater than 10 then\n    say \"big\"\notherwise\n    say \"small\"\nend\nswap a and b\n",
        ),
        (
            "kotlin",
            "fun main() {\n    val x = 5\n    val msg = when (x) {\n        1 -> \"one\"\n        in 2..5 -> \"few\"\n        else -> \"many\"\n    }\n    println(msg)\n    listOf(1, 2).forEach { println(it) }\n}\ndata class P(val x: Int, val y: Int)\n",
        ),
        (
            "sql",
            "people = [{name: \"Ada\", age: 36}, {name: \"Alan\", age: 41}]\nresult = SELECT name, age FROM people WHERE age > 30 ORDER BY age DESC LIMIT 1\ninsert into people (name, age) values (\"Grace\", 85)\nadults = from p in people where p.age >= 18 orderby p.name select p.name\n",
        ),
        (
            "ml",
            "model XorNet:\n    dense 2 -> 8, tanh\n    dense 8 -> 1, sigmoid\nnet = XorNet()\nX = tensor [[0, 0], [0, 1], [1, 0], [1, 1]]\ntrain net on X, y for 500 epochs using adam(lr: 0.05) minimizing bce\nf = x => x ** 2\nprint(f′(3))\n",
        ),
        (
            "swift",
            "func greet(_ name: String) -> String {\n    return \"Hello, \\(name)\"\n}\nlet nums = [3, 1, 2].sorted { $0 < $1 }\nswitch nums.count {\ncase 0: print(\"none\")\ncase 1...3: print(\"few\")\ndefault: print(\"many\")\n}\nenum Direction { case north, south }\n",
        ),
        (
            "mixed",
            "def square(x) { x * x }\nfn cube(x) -> int: return x ** 3\nfunction quad(x) = x ** 4\nprint square 3\nputs [1, 2, 3].map { |x| x + 1 }\nlet result = if 3 > 2 then \"yes\" else \"no\" end\nvalue = match 5 { 0 => \"z\", n => \"n\" }\nprint(f\"{value!r:>10} and {result}\")\n",
        ),
        (
            "python-class",
            "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    @property\n    def size(self):\n        return len(self.items)\ntry:\n    x = 1 / 0\nexcept ZeroDivisionError as e:\n    print('oops', e)\nfinally:\n    print('done')\nwith open('f') as fh:\n    pass\nsquares = {x: x * x for x in range(5) if x % 2 == 0}\n",
        ),
        (
            "misc",
            "x = [1, 2, 3][1:]\ny = x[::-1]\nz = a ?? b ?: c\nw = not a and b or c\nv = 1 < x <= 10\nu = xs |> map(fn(x) => x + 1) |> sum\nt = 2x + 3\ns = √16\nr = n!\nq = 1 to 10 by 2\np = the square root of 16\nn is even\n",
        ),
    ];
    let mut ok = true;
    for (name, src) in cases {
        ok &= check(name, src);
    }
    assert!(ok, "some dialects failed to parse");
}
