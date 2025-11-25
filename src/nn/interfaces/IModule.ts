import type {NewTensor} from "../tensor_new.ts";

export interface IModule {
    forward(input: NewTensor): NewTensor;
}