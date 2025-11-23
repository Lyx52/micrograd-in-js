import type {Tensor} from "./tensor.ts";

export interface ICallable {
    execute(inputs: Tensor): Tensor;
}