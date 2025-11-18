import './style.css'
import {renderGraph} from "./nn/utils.ts";
import {Tensor} from "./nn/tensor.ts";
import {Layer} from "./nn/layer.ts";
import {MLP} from "./nn/mlp.ts";

const mlp = new MLP(4, [4, 1]);
const res = mlp.execute(Tensor.fromValues(6, 3, 4, 5))
const result = res.item().shift();
result.backward();

renderGraph(result);
//runSimpleMLP();
// //
// // const a = new Value(5)
// // const b = new Value(4)
// // const c = new Value(10)
// // const res = c.mul(a.add(b));
// // res.backward();
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
