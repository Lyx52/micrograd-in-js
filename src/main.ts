import {runMinst} from "./tests/minst.ts";
import {NewTensor} from "./nn/tensor_new.ts";
const tensor = NewTensor.randn(2, 2, 2)
console.log(tensor);