import type { ICallable } from "./ICallable";
import type {Tensor} from "./tensor.ts";

export class Softmax implements ICallable {
    execute(inputs: Tensor): Tensor {
        return inputs.softmax();
    }
}