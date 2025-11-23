import type { ICallable } from "./ICallable";
import type {Tensor} from "./tensor.ts";

export class ReLu implements ICallable {
    execute(inputs: Tensor): Tensor {
        return inputs.relu();
    }
}