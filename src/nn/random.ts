
export class RandomGenerator {
    public seed: number = -1;
    private modulus: number = -1;
    private a: number = -1;
    private c: number = -1;
    protected static _instance: RandomGenerator = null;
    constructor(seed = Date.now(), m = Math.pow(2, 31) + 1, a = 22695477, c = 1) {
        this.seed = seed;
        this.modulus = m;
        this.a = a;
        this.c = c;
    }

    public static Instance(): RandomGenerator {
        if (this._instance) return this._instance;

        this._instance = new RandomGenerator();
        return this._instance;
    }

    public static Seed(seed: number):  RandomGenerator {
        this._instance = new RandomGenerator(seed);
        this.random();
        return this._instance;
    }

    public static random() {
        const instance = RandomGenerator.Instance();
        const result = (instance.a * instance.seed + instance.c) % instance.modulus;
        instance.seed = result;

        return result / instance.c;
    }
}