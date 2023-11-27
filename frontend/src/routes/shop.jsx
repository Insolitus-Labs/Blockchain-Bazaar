import Table from "../components/Table";

import './shop.css';
import testData from "../testData.json";

const columns = [
  { label: "Item", accessor: "item", sortable: true },
  { label: "Price", accessor: "price", sortable: true },
  { label: "Quantity", accessor: "quantity", sortable: true },
]

export default function App() {
  return (
    <div className='main-app'>
      <div className="table_container">
        <h2>Available Items</h2>
        <Table data={testData} columns={columns}/>
      </div>
    </div>
  );
}
