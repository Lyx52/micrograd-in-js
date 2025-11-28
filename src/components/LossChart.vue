<script lang="ts">
import {defineComponent} from 'vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import { Line } from 'vue-chartjs'

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

export default defineComponent({
  name: "LossChart",
  components: {
    Line
  },
  props: {
    loss: Array<number>
  },
  computed: {
    chartData() {
      const labels = this.loss.map((v, i) => (i + 1).toString());

      return {
        labels: labels,
        datasets: [
          {
            label: 'Loss',
            backgroundColor: '#f87979',
            data: this.loss,
          }
        ]
      }
    }
  },
  data() {
    return {
      chartOptions: {
        responsive: true,
        maintainAspectRatio: false
      },
    }
  }
})
</script>

<template>
  <div>
    <Line :data="chartData" :options="chartOptions" />
  </div>
</template>

<style scoped>

</style>