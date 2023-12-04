import { useEffect } from 'react';
import { useState } from 'react';

import Table from "../components/Table";

import '../css/shop.css';

const columns = [
  { label: "Item", accessor: "item", sortable: true },
  { label: "Level", accessor: "level", sortable: true },
  { label: "Rarity", accessor: "rarity", sortable: true },
  { label: "Weight", accessor: "weight", sortable: true },
  { label: "Damage", accessor: "damage", sortable: true },
  { label: "Defense", accessor: "defense", sortable: true },
  { label: "Range", accessor: "range", sortable: true },
  { label: "Speed", accessor: "speed", sortable: true },
  { label: "Buy", accessor: "buy_price", sortable: true },
  { label: "Sell", accessor: "sell_price", sortable: true },
]

export default function Shop() {
  const [loading, setLoading] = useState(false);
  const [tableData, setTableData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      const response = await fetch("https://bazzar.mooo.com/get_all_items");
      const data = await response.json();
      setTableData(data);
      setLoading(false);
    };

    fetchData();
  }, [])

  return (
    <div className='main-app'>
      <div className="table_container">
        <h2>Available Items</h2>
        {loading ? <h4>Loading...</h4> : <Table data={tableData} columns={columns}/>}
      </div>
    </div>
  );
}
