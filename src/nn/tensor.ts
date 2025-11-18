import {Value} from "./value.ts";

export type TensorDimension = (Value|TensorDimension)[];
export type TensorApplyCallback = (value: Value, index: number) => Value;
const getTotalElements = (...dims: number[]): number => {
    let elements = dims[0];
    for (let i = 0; i < dims.length - 1; i++) {
        elements *= dims[i + 1];
    }

    return elements;
}

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

const flatten = (dimension: TensorDimension): TensorDimension => {
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

    public item(): Value[] {
        return this.backing as Value[];
    }

    public clone() {
        const tensor = new Tensor(...this.dimensions)

        return tensor.apply((value: Value, index: number) => {
            return (this.backing[index] as Value).clone();
        })
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

    public view(...dims: number[]): TensorDimension {
        const viewElements = getTotalElements(...dims);
        if (viewElements !== this.elements) {
            throw new Error(`View has ${viewElements} elements while expected ${this.elements} elements`);
        }

        const [result, _] = makeView(this.backing, dims);

        return result;
    }

    public sum(): Tensor {
        return Tensor.from(Value.Sum(...this.backing as Value[]));
    }

    public tanh(): Tensor {
        return this.apply((value: Value) => {
            return value.tanh();
        });
    }

    public mse(expected: Tensor|number): Tensor {
        return this.apply((value: Value, index: number)=> {
            if (expected instanceof Tensor) {
                return Value.Mse(expected.Backing[index] as Value, value);
            }

            return Value.Mse(expected as number, value);
        });
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

    public mul(other: Value|Tensor) {
        if (other instanceof Tensor) {
            if (!this.equalDimensions(other)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${other.dimensions.join(', ')}]`);
            }

            const tensor = this.clone();
            return tensor.apply((value: Value, index: number) => {
                return value.mul(other.backing[index] as Value)
            });
        }

        for (let i = 0; i < this.backing.length; i++) {
            this.backing[i] = (this.backing[i] as Value).mul(other as Value);
        }

        return this;
    }

    public add(other: Value|Tensor) {
        if (other instanceof Tensor) {
            if (!this.equalDimensions(other)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${other.dimensions.join(', ')}]`);
            }

            const tensor = this.clone();
            return tensor.apply((value: Value, index: number) => {
                return value.add(other.backing[index] as Value)
            });
        }

        for (let i = 0; i < this.backing.length; i++) {
            this.backing[i] = (this.backing[i] as Value).add(other as Value);
        }

        return this;
    }

    public sub(other: Value|Tensor) {
        if (other instanceof Tensor) {
            if (!this.equalDimensions(other)) {
                throw new Error(`Other tensor dimensions do not match [${this.dimensions.join(', ')}] and [${other.dimensions.join(', ')}]`);
            }

            const tensor = this.clone();
            return tensor.apply((value: Value, index: number) => {
                return value.sub(other.backing[index] as Value)
            });
        }

        for (let i = 0; i < this.backing.length; i++) {
            this.backing[i] = (this.backing[i] as Value).sub(other as Value);
        }

        return this;
    }

    public static randn(...dims: number[]): Tensor {
        const tensor = new  Tensor(...dims);

        return tensor.apply(() => {
            return Value.Random();
        });
    }
}