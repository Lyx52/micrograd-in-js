import {Value} from "./value.ts";
import {getTotalElements} from "../nn/utils.ts";

export type TensorDimension = (Value|TensorDimension)[];
export type TensorApplyCallback = (value: Value, index: number) => Value;
export type TensorCollapseCallback = (value: Value[], index: number) => Value;


const getLength = (dimension: TensorDimension, target: number, current: number = 0): number => {
    if (target === current) {
        return dimension.length;
    }

    return getLength(dimension, target, current + 1);
}

const getDimensions = (dimension: TensorDimension|Value, dims: number[] = []): number[] => {
    if (Array.isArray(dimension)) {
        const dim = dimension.length;
        dims.push(dim);
        getDimensions(dimension[0], dims);
    }

    return dims;
}

export const flatten = (dimension: TensorDimension): TensorDimension => {
    if (Array.isArray(dimension)) {
        return dimension.flatMap(v => flatten(v as TensorDimension[]));
    }

    return dimension;
}

function makeView(elements: TensorDimension, dims: number[], currentIndex: number = 0, currentLevel: number = 0): [TensorDimension, number] {
    const levels = dims.length - 1;
    const buffer: TensorDimension = [];
    if (currentLevel === levels) {
        return [elements.slice(currentIndex, currentIndex + dims[currentLevel]), currentIndex + dims[currentLevel]];
    }

    for (let i = 0; i < dims[currentLevel]; i++) {
        const [view, index] = makeView(elements, dims, currentIndex,  currentLevel + 1);
        currentIndex = index;
        buffer.push(view);
    }

    return [buffer, currentIndex];
}

export function collapseDimensions(elements: TensorDimension, combined: number[], collapseCallback: TensorCollapseCallback): TensorDimension {
    const dims = getDimensions(elements);
    const current = combined.shift();
    
    const result: TensorDimension[] = [];


    console.log(dims)
    return elements;
}

export class Tensor {
    private backing: TensorDimension;
    private dimensions: number[];
    private elements: number;
    constructor(...dims: number[]) {
        if (dims.length === 0) {
            throw new Error('Tensor must be atleast one dimensional');
        }

        this.dimensions = [...dims];
        this.elements = getTotalElements(...dims);
        this.backing = [];
        for (let i = 0; i < this.elements; i++) {
            this.backing.push(new Value(0));
        }
    }

    public setBacking(backing: TensorDimension): Tensor {
        this.backing = backing;

        return this;
    }

    public gradients() {
        return this.item().map(v => v.gradients())
    }

    public setDimensions(dims: number[]): Tensor {
        this.dimensions = dims;

        return this;
    }

    public static fromValues(...values: number[]) {
        return this.from(...values.map(v => new Value(v)));
    }

    public static fromTensors(...values: Tensor[]) {
        const data = [];
        for (const tensor of values) {
            data.push(...tensor.backing);
        }

        return Tensor.from(...data)
    }

    public static from(...values: TensorDimension) {
        const dims = getDimensions(values);
        const tensor = new Tensor(...dims);
        tensor.backing = flatten(values);
        return tensor;
    }

    public isScalar() {
        return this.dimensions.length === 1 && this.dimensions[0] === 1;
    }

    public scalar(): Value {
        if (!this.isScalar()) {
            throw new Error('Tensor must be a scalar value!');
        }

        return this.backing[0] as Value;
    }

    public item(): Value[] {
        return this.backing as Value[];
    }

    public clone() {
        const tensor = new Tensor(...this.dimensions)
        tensor.backing = this.item().map(v => v.clone());

        return tensor;
    }

    public apply(callback: TensorApplyCallback): Tensor {
        for (let i = 0; i < this.backing.length; i++) {
            this.backing[i] = callback(this.backing[i] as Value, i);
        }

        return this;
    }

    public get Backing(): TensorDimension {
        return this.backing;
    }

    public length(dimension: number = 0): number {
        return getLength(this.backing, dimension, this.elements);
    }

    public repr(): TensorDimension {
        return this.view(...this.dimensions);
    }

    public flatten(): Tensor {
        return Tensor.from(...this.item());
    }

    public relu(): Tensor {
        const tensor = new Tensor(...this.dimensions);

        return tensor.apply((_: Value, index: number) => {
            return (this.backing[index] as Value).clone().relu();
        });
    }

    public view(...dims: number[]): TensorDimension {
        const viewElements = getTotalElements(...dims);
        if (viewElements !== this.elements) {
            throw new Error(`View has ${viewElements} elements while expected ${this.elements} elements`);
        }

        const [result, _] = makeView(this.backing, dims);

        return result;
    }

    public equalDimensions(other: Tensor): boolean {
        if (other.dimensions.length !== this.dimensions.length) {
            return false;
        }

        for (let i = 0; i < this.dimensions.length; i++) {
            if (this.dimensions[i] !== other.dimensions[i]) {
                return false;
            }
        }

        return true;
    }

    public sum(): Tensor {
        return Tensor.from(Value.Sum(...this.item()));
    }

    public tanh(): Tensor {
        return Tensor.from(this.item()
            .map(v => v.tanh())
        );
    }

    public softmax(dim: number = 0): Tensor {
        const sum = Value.Sum(...this.item());
        return Tensor.from(this.item()
            .map(v => v.div(sum))
        );
    }

    public mse(expected: Value|number|Tensor): Tensor {
        if (expected instanceof Tensor) {
            return Tensor.from(this.item()
                .map((v, i) => v.mse(expected.item()[i]))
            );
        }

        return Tensor.from(this.item()
            .map(v => v.mse(expected as Value|number))
        );
    }

    public mul(other: Value|number|Tensor) {
        if (other instanceof Tensor) {
            if (!this.equalDimensions(other)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${other.dimensions.join(', ')}]`);
            }

            return Tensor.from(this.item()
                .map((v, i) => v.mul(other.item()[i]))
            )
        }

        return Tensor.from(this.item()
            .map(v => v.mul(other as Value|number))
        );
    }

    public div(other: Value|number|Tensor) {
        if (other instanceof Tensor) {
            if (!this.equalDimensions(other)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${other.dimensions.join(', ')}]`);
            }

            return Tensor.from(this.item()
                .map((v, i) => v.div(other.item()[i]))
            )
        }

        return Tensor.from(this.item()
            .map(v => v.div(other as Value|number))
        );
    }

    public add(other: Value|number|Tensor) {
        if (other instanceof Tensor) {
            if (!this.equalDimensions(other)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${other.dimensions.join(', ')}]`);
            }

            return Tensor.from(this.item()
                .map((v, i) => v.add(other.item()[i]))
            )
        }

        return Tensor.from(this.item()
            .map(v => v.add(other as Value|number))
        );
    }

    public sub(other: Value|number|Tensor) {
        if (other instanceof Tensor) {
            if (!this.equalDimensions(other)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${other.dimensions.join(', ')}]`);
            }

            return Tensor.from(this.item()
                .map((v, i) => v.sub(other.item()[i]))
            )
        }

        return Tensor.from(this.item()
            .map(v => v.sub(other as Value|number))
        );
    }

    public pow(other: Value|number|Tensor) {
        if (other instanceof Tensor) {
            if (!this.equalDimensions(other)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${other.dimensions.join(', ')}]`);
            }

            return Tensor.from(this.item()
                .map((v, i) => v.pow(other.item()[i]))
            )
        }

        return Tensor.from(this.item()
            .map(v => v.pow(other as Value|number))
        );
    }

    public static randn(...dims: number[]): Tensor {
        const tensor = new  Tensor(...dims);

        return tensor.apply(() => {
            return Value.Random();
        });
    }
}