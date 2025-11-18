import type { Value } from "./value";
import type {Tensor} from "./tensor.ts";

export interface ICallable {
    execute(inputs: Tensor): Tensor;
    parameters(): Value[];
    zerograd();
}