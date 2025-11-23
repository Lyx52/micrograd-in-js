import type { ICallable } from "./ICallable";
import type {Tensor} from "./tensor.ts";

export class Flatten implements ICallable {
    execute(inputs: Tensor): Tensor {
        return inputs.flatten();
    }
}