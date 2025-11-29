import {RandomGenerator} from "../../nn/random.ts";
import {NewTensor} from "../../nn/tensor_new.ts";
import {Tensor} from "../tensor.ts";
export const testNewVsOldTensor = () => {
    RandomGenerator.Seed(0);
    console.time('NewTensorMulBackward')
    const a = NewTensor.randn(64, 64, 64);
    a.mul(32).sub(-1).pow(2);
    a.backward();
    console.timeEnd('NewTensorMulBackward')

    console.time('TensorMulBackward')
    const b = Tensor.randn(64, 64, 64);
    b.mul(32).sub(-1).pow(2);
    for (const val of b.item()) {
        val.backward();
    }
    console.timeEnd('TensorMulBackward')
    console.log(b, a);
}
