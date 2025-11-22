import {renderGraph} from "./nn/utils.ts";
import {Tensor} from "./nn/tensor.ts";
import {Layer} from "./nn/layer.ts";
import {MLP} from "./nn/mlp.ts";
import {Value} from "./nn/value.ts";
import {runSimpleMLP} from "./tests/simple_mlp.ts";

// const mlp = new MLP(4, [4, 1]);
// const res = mlp.execute(Tensor.fromValues(6, 3, 4, 5))
// const result = res.item().shift();
// result.backward();


runSimpleMLP();
// //
// const a = new Value(5)
// const b = new Value(4)
// const c = new Value(10)
// const res = Value.Mse(3, c.mul(a.add(b))).tanh();
//
// res.backward();
// renderGraph(res);
// const a1 = Tensor.fromValues(5)
// const b1 = Tensor.fromValues(4)
// const c1 = Tensor.fromValues(10)
// const res1 = c1.mul(a1.add(b1)).mse(Tensor.fromValues(3)).tanh();
// const item = res1.item();
//
// item[0].backward();
// renderGraph(item[0]);
// //
// // renderGraph(res);
// const a = Tensor.randn(3, 2, 3, 2);
// const b = Tensor.randn(3, 2, 3, 2);
// const c = Tensor.randn(3, 3);
//
// const res = a.sub(b).sum().tanh();
//
//
// res.backward();
// renderGraph(res)
// console.log(res);
